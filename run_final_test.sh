#!/usr/bin/env bash
set -euo pipefail

# 中文说明：supermarket_sorting_final 的宿主机总启动器。
# 它负责校验参数、选择/固定测试布局、启动 Server 与 Client 容器、抓取任务
# 消息、启动 RViz，并把所有输出保存到一次运行独立的日志目录。
# 本脚本独立于旧的 run_nav2_test.sh，后者仍启动历史固定可乐场景。

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${WORKSPACE_ROOT}/supermarket_sorting_final"
SERVER_IMAGE="${SUPERMARKET_SERVER_IMAGE:-supermarket_sorting:server}"
CLIENT_IMAGE="${SUPERMARKET_CLIENT_IMAGE:-supermarket_sorting:client-nav2}"
SERVER_CONTAINER="supermarket_sorting_final_server"
CLIENT_CONTAINER="supermarket_sorting_final_client"
CACHE_VOLUME="supermarket_sorting_cache"
PRODUCT_PT_WEIGHT_FILE="${PROJECT_ROOT}/weights/products.pt"
PRODUCT_WEIGHT_FILE="$PRODUCT_PT_WEIGHT_FILE"
PRODUCT_WEIGHT_BACKEND="PyTorch"

RUN_MODE="${RUN_MODE:-nav_only}"
CONTROLLER="${CONTROLLER:-mppi}"
ENABLE_RANDOM_OBSTACLES="${ENABLE_RANDOM_OBSTACLES:-1}"
FULL_ARM_CLEARANCE_VERIFIED="${FULL_ARM_CLEARANCE_VERIFIED:-0}"
SUPERMARKET_E_THREE_KELE_DEBUG="${SUPERMARKET_E_THREE_KELE_DEBUG:-0}"
SUPERMARKET_E_L3_RIGHT_KELE_DEBUG="${SUPERMARKET_E_L3_RIGHT_KELE_DEBUG:-0}"
SUPERMARKET_E_FIVE_KELE_DEBUG="${SUPERMARKET_E_FIVE_KELE_DEBUG:-0}"
SUPERMARKET_E_THREE_MAIDONG_DEBUG="${SUPERMARKET_E_THREE_MAIDONG_DEBUG:-0}"
SUPERMARKET_E_MIXED_DEBUG="${SUPERMARKET_E_MIXED_DEBUG:-0}"
SUPERMARKET_E_MIDDLE_FOOD_DEBUG="${SUPERMARKET_E_MIDDLE_FOOD_DEBUG:-0}"
SUPERMARKET_E_MIDDLE_GUM_FRUIT_DEBUG="${SUPERMARKET_E_MIDDLE_GUM_FRUIT_DEBUG:-0}"
SUPERMARKET_E_FIVE_KIND_DEBUG="${SUPERMARKET_E_FIVE_KIND_DEBUG:-0}"
SUPERMARKET_E_MIDDLE_TISSUE_DEBUG="${SUPERMARKET_E_MIDDLE_TISSUE_DEBUG:-0}"
SUPERMARKET_E_TISSUE_FIVE_KIND_DEBUG="${SUPERMARKET_E_TISSUE_FIVE_KIND_DEBUG:-0}"
SUPERMARKET_E_MIXED_SEQUENCE="${SUPERMARKET_E_MIXED_SEQUENCE:-maidong,kele,maidong}"
SUPERMARKET_FORCE_FIRST_TARGET_KIND="${SUPERMARKET_FORCE_FIRST_TARGET_KIND:-}"
TASK_SEED="${TASK_SEED:-$(date +%s%N)}"
if [[ ! "$TASK_SEED" =~ ^-?[0-9]+$ ]]; then
  echo "ERROR: TASK_SEED must be an integer (got: $TASK_SEED)" >&2
  exit 2
fi
SUPERMARKET_INVENTORY_MIN_VOTES="${SUPERMARKET_INVENTORY_MIN_VOTES:-3}"
SUPERMARKET_SCAN_STAND_OFF="${SUPERMARKET_SCAN_STAND_OFF:-0.85}"
SUPERMARKET_APPROACH_STAND_OFF="${SUPERMARKET_APPROACH_STAND_OFF:-0.70}"
# 这组临时固定任务需要复用正式随机调度里的首次 E 柜直控高速段。
# 在计算扫描高度和后续校验前切换模式，避免仍按 e_mixed_cycle 向 E 观察点发 Nav2 目标。
if [[ "$SUPERMARKET_E_TISSUE_FIVE_KIND_DEBUG" == "1" ]]; then
  RUN_MODE="random_cycle"
fi
if [[ "$RUN_MODE" == "e_kele_cycle" || "$RUN_MODE" == "e_maidong_cycle" \
    || "$RUN_MODE" == "e_mixed_cycle" || "$RUN_MODE" == "random_cycle" ]]; then
  # One fixed observation height: all three E-row centres stay inside the
  # 45.29-degree head-camera FOV without moving the torso during observation.
  SUPERMARKET_SCAN_SLIDE="${SUPERMARKET_SCAN_SLIDE:-0.30}"
else
  SUPERMARKET_SCAN_SLIDE="${SUPERMARKET_SCAN_SLIDE:-0.15}"
fi

# These body IDs are documented in the public item list.  The server randomizes
# their shelf positions before publishing anonymous IDs.  In random_cycle,
# TASKS names five distinct physical items; their product classes may repeat.
debug_layout_count=0
for debug_layout_flag in \
  "$SUPERMARKET_E_THREE_KELE_DEBUG" \
  "$SUPERMARKET_E_L3_RIGHT_KELE_DEBUG" \
  "$SUPERMARKET_E_FIVE_KELE_DEBUG" \
  "$SUPERMARKET_E_THREE_MAIDONG_DEBUG" \
  "$SUPERMARKET_E_MIXED_DEBUG" \
  "$SUPERMARKET_E_MIDDLE_FOOD_DEBUG" \
  "$SUPERMARKET_E_MIDDLE_GUM_FRUIT_DEBUG" \
  "$SUPERMARKET_E_FIVE_KIND_DEBUG" \
  "$SUPERMARKET_E_MIDDLE_TISSUE_DEBUG" \
  "$SUPERMARKET_E_TISSUE_FIVE_KIND_DEBUG"; do
  if [[ "$debug_layout_flag" == "1" ]]; then
    debug_layout_count=$((debug_layout_count + 1))
  fi
done
if (( debug_layout_count > 1 )); then
  echo "ERROR: enable only one E-shelf debug layout" >&2
  exit 2
fi

# Generate five distinct physical bodies from the complete catalogue.  Product
# classes may repeat because every class owns five bodies.  When at least one
# tissue body is selected, move only the first tissue to the head of the
# published task; any additional tissue bodies retain their relative order and
# are handled like the remaining four deliveries.
generate_random_five_item_task() {
  python3 - "$TASK_SEED" <<'PY'
import random
import sys

seed = int(sys.argv[1])
rng = random.Random(seed)
product_ids_by_kind = {
    "sanmingzhi": (1, 10, 19, 28, 37),
    "heweidao": (2, 11, 20, 29, 38),
    "shupian": (3, 12, 21, 30, 39),
    "zhijin": (4, 13, 22, 31, 40),
    "maidong": (5, 6, 14, 23, 41),
    "kele": (15, 24, 32, 33, 42),
    "kouxiangtang": (7, 16, 25, 34, 43),
    "pingguo": (8, 17, 26, 35, 44),
    "chengzi": (9, 18, 27, 36, 45),
}
kind_by_id = {
    product_id: kind
    for kind, product_ids in product_ids_by_kind.items()
    for product_id in product_ids
}
selected_ids = rng.sample(range(1, 46), 5)
first_tissue = next(
    (
        index
        for index, product_id in enumerate(selected_ids)
        if kind_by_id[product_id] == "zhijin"
    ),
    None,
)
if first_tissue is not None:
    selected_ids.insert(0, selected_ids.pop(first_tissue))
selected_kinds = [kind_by_id[product_id] for product_id in selected_ids]
tasks = ",".join(f"product_{product_id:03d}" for product_id in selected_ids)
sequence = ",".join(selected_kinds)
forced_first = "zhijin" if first_tissue is not None else ""
print(f"{tasks}|{sequence}|{forced_first}")
PY
}

if [[ "$SUPERMARKET_E_TISSUE_FIVE_KIND_DEBUG" == "1" ]]; then
  # Compatibility layout 10 retains its fixed shelf/obstacle reproducibility,
  # but the five published products now use the same formal random contract.
  IFS='|' read -r TASKS SUPERMARKET_E_MIXED_SEQUENCE \
    SUPERMARKET_FORCE_FIRST_TARGET_KIND < <(generate_random_five_item_task)
  PRODUCT_SEED=21725
elif [[ "$SUPERMARKET_E_MIDDLE_TISSUE_DEBUG" == "1" ]]; then
  # Compatibility layout 9: keep the validated E/L2/C2 target and four filler
  # bodies required by the five-ID Server interface.
  TASKS="product_004,product_001,product_002,product_003,product_005"
  PRODUCT_SEED=89
  SUPERMARKET_E_MIXED_SEQUENCE="zhijin"
elif [[ "$SUPERMARKET_E_MIDDLE_GUM_FRUIT_DEBUG" == "1" ]]; then
  # Server shuffle seed 493 puts the three new targets across E's middle row:
  # kouxiangtang=L2/C1, pingguo=L2/C2 and chengzi=L2/C3.  The two additional
  # task bodies are non-target fillers because the Server order interface
  # requires exactly five distinct product IDs.
  TASKS="product_025,product_035,product_027,product_001,product_002"
  PRODUCT_SEED=493
  # Exercise the orange-specific C3 approach first on the next regression;
  # gum and apple retain their proven positions and follow afterward.
  SUPERMARKET_E_MIXED_SEQUENCE="chengzi,kouxiangtang,pingguo"
elif [[ "$SUPERMARKET_E_FIVE_KIND_DEBUG" == "1" ]]; then
  # Server shuffle seed 349 puts exactly one of every requested class on
  # shelf E, spreads the targets over all three rows and columns, and leaves
  # the other four E slots free of duplicate target classes:
  # pingguo=L1/C2, sanmingzhi=L2/C1, chengzi=L2/C2,
  # heweidao=L2/C3 and kouxiangtang=L3/C1.
  # TASKS and the mission sequence follow the requested pick order below.
  TASKS="product_018,product_017,product_007,product_038,product_019"
  PRODUCT_SEED=349
  SUPERMARKET_E_MIXED_SEQUENCE="chengzi,pingguo,kouxiangtang,heweidao,sanmingzhi"
elif [[ "$SUPERMARKET_E_MIXED_DEBUG" == "1" ]]; then
  # Server shuffle seed 423 places the requested mixed order on shelf E:
  # product_041 maidong -> L1/C1 (ArUco 36), product_042 kele -> L2/C2
  # (ArUco 40), product_023 maidong -> L3/C2 (ArUco 43).  TASKS order is
  # deliberately preserved as maidong -> kele -> maidong.
  TASKS="product_041,product_042,product_023,product_001,product_002"
  PRODUCT_SEED=423
  SUPERMARKET_E_MIXED_SEQUENCE="maidong,kele,maidong"
elif [[ "$SUPERMARKET_E_MIDDLE_FOOD_DEBUG" == "1" ]]; then
  # Server shuffle seed 2151 places exactly the three requested food packages
  # on E's middle row: sanmingzhi=L2/C1 (ArUco 39), heweidao=L2/C2
  # (ArUco 40), shupian=L2/C3 (ArUco 41).  Put heweidao first in this regression
  # run, followed by shupian and sanmingzhi.
  TASKS="product_011,product_039,product_028,product_004,product_005"
  PRODUCT_SEED=2151
  SUPERMARKET_E_MIXED_SEQUENCE="heweidao,shupian,sanmingzhi"
elif [[ "$SUPERMARKET_E_THREE_MAIDONG_DEBUG" == "1" ]]; then
  # Server shuffle seed 27 deterministically puts exactly three maidong
  # bottles on shelf E, one per row: product_005->L1/C1,
  # product_006->L2/C2 and product_023->L3/C3.  The two filler task bodies
  # are non-maidong, so clear_e_maidong requests exactly three deliveries.
  TASKS="product_005,product_006,product_023,product_001,product_002"
  PRODUCT_SEED=27
elif [[ "$SUPERMARKET_E_FIVE_KELE_DEBUG" == "1" ]]; then
  # Server shuffle seed 447514 puts all five kele bodies on shelf E, spread
  # across L1/C1+C3, L2/C1 and L3/C1+C3. Request all five so this layout
  # exercises five consecutive observe/pick/deliver cycles.
  TASKS="product_015,product_024,product_032,product_033,product_042"
  PRODUCT_SEED=447514
elif [[ "$SUPERMARKET_E_L3_RIGHT_KELE_DEBUG" == "1" ]]; then
  # Server shuffle seed 48 places the only E-shelf kele at L3/C3:
  # product_042->E/L3/C3 (ArUco 44).  The other four kele bodies are shuffled
  # onto shelves B-D, and only product_042 is requested by this test order.
  TASKS="product_042,product_001,product_002,product_003,product_004"
  PRODUCT_SEED=48
elif [[ "$SUPERMARKET_E_THREE_KELE_DEBUG" == "1" ]]; then
  # Server shuffle seed 11 deterministically places exactly one kele in every
  # E row: product_015->L1/C1, product_033->L2/C3, product_042->L3/C3.
  # Request those three bodies so the cycle exercises all three grasp heights.
  TASKS="product_015,product_033,product_042,product_001,product_002"
  PRODUCT_SEED=11
else
  if [[ "$RUN_MODE" == "random_cycle" && -z "${TASKS:-}" ]]; then
    # Formal random mission: five physical bodies, all nine classes eligible,
    # repeated classes allowed.  Paper is promoted to the first task position.
    IFS='|' read -r TASKS SUPERMARKET_E_MIXED_SEQUENCE \
      SUPERMARKET_FORCE_FIRST_TARGET_KIND < <(generate_random_five_item_task)
  else
    TASKS="${TASKS:-product_001,product_002,product_003,product_004,product_015}"
  fi
  PRODUCT_SEED="${PRODUCT_SEED:-$(date +%s)}"
fi
if [[ ! "$PRODUCT_SEED" =~ ^-?[0-9]+$ ]]; then
  echo "ERROR: PRODUCT_SEED must be an integer (got: $PRODUCT_SEED)" >&2
  exit 2
fi
# 商品调试模式会故意固定 PRODUCT_SEED，以便每次保持相同货架布局。
# 障碍物使用独立的纳秒级种子。原来商品和障碍默认都读取同一秒的时间，
# 两者经常得到相同种子，并且在一秒内快速重启会复现同一障碍布局。
# 显式传入 OBSTACLE_SEED 时仍可准确复现某一轮场景。
OBSTACLE_SEED="${OBSTACLE_SEED:-$(date +%s%N)}"
if [[ ! "$OBSTACLE_SEED" =~ ^-?[0-9]+$ ]]; then
  echo "ERROR: OBSTACLE_SEED must be an integer (got: $OBSTACLE_SEED)" >&2
  exit 2
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)_p${PRODUCT_SEED}_o${OBSTACLE_SEED}}"
LOG_BASE="${LOG_BASE:-${PROJECT_ROOT}/logs}"
LOG_DIR="${LOG_DIR:-${LOG_BASE}/${RUN_ID}}"
SELF="$(readlink -f "${BASH_SOURCE[0]}")"
ROLE="${1:-}"


usage() {
  cat <<'EOF'
Usage:
  ./run_final_test.sh [--check|--stop]

Common environment variables:
  RUN_MODE=nav_only|scan_only|full|e_kele_cycle|e_maidong_cycle|e_mixed_cycle|random_cycle
                                            default: nav_only
  CONTROLLER=dwb|mppi                    default: mppi
  ENABLE_RANDOM_OBSTACLES=0|1            default: 1
  PRODUCT_SEED=<integer>                 default: current Unix time
  TASK_SEED=<integer>                    random_cycle order seed
  OBSTACLE_SEED=<integer>                default: current high-resolution time
  TASKS=product_001,...,product_005       exactly five distinct product_NNN IDs;
                                            classes may repeat in random_cycle
  FULL_ARM_CLEARANCE_VERIFIED=0|1        required for arm-cycle modes
  SUPERMARKET_E_THREE_KELE_DEBUG=0|1    seed E with one kele in each row
  SUPERMARKET_E_L3_RIGHT_KELE_DEBUG=0|1 seed only E/L3/C3 with one kele
  SUPERMARKET_E_FIVE_KELE_DEBUG=0|1     seed E with L1=2, L2=1, L3=2 kele
  SUPERMARKET_E_THREE_MAIDONG_DEBUG=0|1 seed E with one maidong in each row
  SUPERMARKET_E_MIXED_DEBUG=0|1         E: L1 maidong, L2 kele, L3 maidong
  SUPERMARKET_E_MIDDLE_FOOD_DEBUG=0|1   E/L2: sanmingzhi, heweidao, shupian
  SUPERMARKET_E_MIDDLE_GUM_FRUIT_DEBUG=0|1
                                            E/L2: kouxiangtang, pingguo, chengzi
  SUPERMARKET_E_FIVE_KIND_DEBUG=0|1     E: one each chengzi, pingguo,
                                            kouxiangtang, heweidao and sanmingzhi
  SUPERMARKET_E_MIDDLE_TISSUE_DEBUG=0|1 compatibility regression layout 9
  SUPERMARKET_E_TISSUE_FIVE_KIND_DEBUG=0|1 compatibility regression layout 10
  SUPERMARKET_INVENTORY_MIN_VOTES=<int>  default: 3
  SUPERMARKET_SCAN_STAND_OFF=<metres>    default: 0.85
  SUPERMARKET_APPROACH_STAND_OFF=<metres> default: 0.70
  SUPERMARKET_SCAN_SLIDE=<metres>        default: 0.15; E cycles: 0.30
  LOG_BASE=<host-directory>              default: supermarket_sorting_final/logs

Examples:
  ./run_final_test.sh --check
  RUN_MODE=nav_only PRODUCT_SEED=20260901 OBSTACLE_SEED=21260904 ./run_final_test.sh
  RUN_MODE=scan_only PRODUCT_SEED=20260901 OBSTACLE_SEED=21260904 ./run_final_test.sh
  RUN_MODE=full FULL_ARM_CLEARANCE_VERIFIED=1 ./run_final_test.sh
  RUN_MODE=e_kele_cycle FULL_ARM_CLEARANCE_VERIFIED=1 SUPERMARKET_SCAN_SLIDE=0.30 ./run_final_test.sh
  RUN_MODE=e_kele_cycle FULL_ARM_CLEARANCE_VERIFIED=1 SUPERMARKET_E_THREE_KELE_DEBUG=1 ./run_final_test.sh
  RUN_MODE=e_kele_cycle FULL_ARM_CLEARANCE_VERIFIED=1 SUPERMARKET_E_L3_RIGHT_KELE_DEBUG=1 ./run_final_test.sh
  RUN_MODE=e_kele_cycle FULL_ARM_CLEARANCE_VERIFIED=1 SUPERMARKET_E_FIVE_KELE_DEBUG=1 ./run_final_test.sh
  RUN_MODE=e_maidong_cycle FULL_ARM_CLEARANCE_VERIFIED=1 SUPERMARKET_E_THREE_MAIDONG_DEBUG=1 ./run_final_test.sh
  RUN_MODE=e_mixed_cycle FULL_ARM_CLEARANCE_VERIFIED=1 SUPERMARKET_E_MIXED_DEBUG=1 ./run_final_test.sh
  RUN_MODE=e_mixed_cycle FULL_ARM_CLEARANCE_VERIFIED=1 SUPERMARKET_E_MIDDLE_FOOD_DEBUG=1 ./run_final_test.sh
  RUN_MODE=e_mixed_cycle FULL_ARM_CLEARANCE_VERIFIED=1 SUPERMARKET_E_MIDDLE_GUM_FRUIT_DEBUG=1 ./run_final_test.sh
  RUN_MODE=e_mixed_cycle FULL_ARM_CLEARANCE_VERIFIED=1 SUPERMARKET_E_FIVE_KIND_DEBUG=1 ./run_final_test.sh
  RUN_MODE=e_mixed_cycle FULL_ARM_CLEARANCE_VERIFIED=1 SUPERMARKET_E_MIDDLE_TISSUE_DEBUG=1 ./run_final_test.sh
  FULL_ARM_CLEARANCE_VERIFIED=1 SUPERMARKET_E_TISSUE_FIVE_KIND_DEBUG=1 ./run_final_test.sh
  RUN_MODE=random_cycle CONTROLLER=mppi ENABLE_RANDOM_OBSTACLES=1 FULL_ARM_CLEARANCE_VERIFIED=1 ./run_final_test.sh

scan_only first lowers the torso to its camera scanning height, then starts Nav2
and nine-class perception. The complete ArUco map is loaded from fixed odom truth.
EOF
}


trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}


normalize_tasks() {
  local raw="${TASKS//;/,}"
  local token cleaned
  local -a tokens normalized
  local -A seen=()
  IFS=',' read -r -a tokens <<< "$raw"
  for token in "${tokens[@]}"; do
    cleaned="$(trim "$token")"
    if [[ ! "$cleaned" =~ ^product_[0-9]{3}$ ]]; then
      echo "ERROR: TASKS entries must be exact product_NNN IDs (got: $cleaned)" >&2
      echo "Use physical Server IDs; repeated classes need different IDs." >&2
      exit 2
    fi
    if [[ -n "${seen[$cleaned]:-}" ]]; then
      echo "ERROR: TASKS contains duplicate ID: $cleaned" >&2
      exit 2
    fi
    seen["$cleaned"]=1
    normalized+=("$cleaned")
  done
  if (( ${#normalized[@]} != 5 )); then
    echo "ERROR: TASKS must contain exactly five distinct IDs; got ${#normalized[@]}" >&2
    exit 2
  fi
  TASKS="$(IFS=,; printf '%s' "${normalized[*]}")"
}


validate_options() {
  case "$RUN_MODE" in
    nav_only|scan_only|full|e_kele_cycle|e_maidong_cycle|e_mixed_cycle|random_cycle) ;;
    *) echo "ERROR: RUN_MODE must be nav_only, scan_only, full, e_kele_cycle, e_maidong_cycle, e_mixed_cycle or random_cycle" >&2; exit 2 ;;
  esac
  case "$CONTROLLER" in
    dwb|mppi) ;;
    *) echo "ERROR: CONTROLLER must be dwb or mppi" >&2; exit 2 ;;
  esac
  case "$ENABLE_RANDOM_OBSTACLES" in
    0|1) ;;
    *) echo "ERROR: ENABLE_RANDOM_OBSTACLES must be 0 or 1" >&2; exit 2 ;;
  esac
  case "$FULL_ARM_CLEARANCE_VERIFIED" in
    0|1) ;;
    *) echo "ERROR: FULL_ARM_CLEARANCE_VERIFIED must be 0 or 1" >&2; exit 2 ;;
  esac
  case "$SUPERMARKET_E_THREE_KELE_DEBUG" in
    0|1) ;;
    *) echo "ERROR: SUPERMARKET_E_THREE_KELE_DEBUG must be 0 or 1" >&2; exit 2 ;;
  esac
  case "$SUPERMARKET_E_L3_RIGHT_KELE_DEBUG" in
    0|1) ;;
    *) echo "ERROR: SUPERMARKET_E_L3_RIGHT_KELE_DEBUG must be 0 or 1" >&2; exit 2 ;;
  esac
  case "$SUPERMARKET_E_FIVE_KELE_DEBUG" in
    0|1) ;;
    *) echo "ERROR: SUPERMARKET_E_FIVE_KELE_DEBUG must be 0 or 1" >&2; exit 2 ;;
  esac
  case "$SUPERMARKET_E_THREE_MAIDONG_DEBUG" in
    0|1) ;;
    *) echo "ERROR: SUPERMARKET_E_THREE_MAIDONG_DEBUG must be 0 or 1" >&2; exit 2 ;;
  esac
  case "$SUPERMARKET_E_MIXED_DEBUG" in
    0|1) ;;
    *) echo "ERROR: SUPERMARKET_E_MIXED_DEBUG must be 0 or 1" >&2; exit 2 ;;
  esac
  case "$SUPERMARKET_E_MIDDLE_FOOD_DEBUG" in
    0|1) ;;
    *) echo "ERROR: SUPERMARKET_E_MIDDLE_FOOD_DEBUG must be 0 or 1" >&2; exit 2 ;;
  esac
  case "$SUPERMARKET_E_MIDDLE_GUM_FRUIT_DEBUG" in
    0|1) ;;
    *) echo "ERROR: SUPERMARKET_E_MIDDLE_GUM_FRUIT_DEBUG must be 0 or 1" >&2; exit 2 ;;
  esac
  case "$SUPERMARKET_E_FIVE_KIND_DEBUG" in
    0|1) ;;
    *) echo "ERROR: SUPERMARKET_E_FIVE_KIND_DEBUG must be 0 or 1" >&2; exit 2 ;;
  esac
  case "$SUPERMARKET_E_MIDDLE_TISSUE_DEBUG" in
    0|1) ;;
    *) echo "ERROR: SUPERMARKET_E_MIDDLE_TISSUE_DEBUG must be 0 or 1" >&2; exit 2 ;;
  esac
  case "$SUPERMARKET_E_TISSUE_FIVE_KIND_DEBUG" in
    0|1) ;;
    *) echo "ERROR: SUPERMARKET_E_TISSUE_FIVE_KIND_DEBUG must be 0 or 1" >&2; exit 2 ;;
  esac
  if [[ ! "$SUPERMARKET_INVENTORY_MIN_VOTES" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: SUPERMARKET_INVENTORY_MIN_VOTES must be a positive integer" >&2
    exit 2
  fi
  if [[ ! "$SUPERMARKET_SCAN_STAND_OFF" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ERROR: SUPERMARKET_SCAN_STAND_OFF must be a positive number" >&2
    exit 2
  fi
  if [[ ! "$SUPERMARKET_APPROACH_STAND_OFF" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ERROR: SUPERMARKET_APPROACH_STAND_OFF must be a positive number" >&2
    exit 2
  fi
  if [[ ! "$SUPERMARKET_SCAN_SLIDE" =~ ^[0-9]+([.][0-9]+)?$ ]] \
      || ! awk -v value="$SUPERMARKET_SCAN_SLIDE" \
        'BEGIN { exit !(value >= 0.0 && value <= 0.40) }'; then
    echo "ERROR: SUPERMARKET_SCAN_SLIDE must be within 0.00..0.40 m" >&2
    exit 2
  fi
  if [[ "$RUN_MODE" == "full" || "$RUN_MODE" == "e_kele_cycle" \
      || "$RUN_MODE" == "e_maidong_cycle" \
      || "$RUN_MODE" == "e_mixed_cycle" || "$RUN_MODE" == "random_cycle" ]] \
      && [[ "$FULL_ARM_CLEARANCE_VERIFIED" != "1" ]]; then
    echo "ERROR: $RUN_MODE mode remains safety-locked." >&2
    echo "Set FULL_ARM_CLEARANCE_VERIFIED=1 only after validating carried-arm clearance." >&2
    exit 2
  fi
  normalize_tasks
  if [[ "$RUN_MODE" == "random_cycle" ]]; then
    local task_id task_number
    local -a normalized_tasks
    IFS=',' read -r -a normalized_tasks <<< "$TASKS"
    for task_id in "${normalized_tasks[@]}"; do
      task_number=$((10#${task_id#product_}))
      if (( task_number < 1 || task_number > 45 )); then
        echo "ERROR: random_cycle TASKS contains unknown product ID: $task_id" >&2
        exit 2
      fi
    done
  fi
}


verify_host() {
  command -v docker >/dev/null || { echo "ERROR: docker is unavailable" >&2; exit 1; }
  [[ -f "${PROJECT_ROOT}/scripts/run_nav2_mission.sh" ]] || {
    echo "ERROR: incomplete project: ${PROJECT_ROOT}/scripts/run_nav2_mission.sh" >&2
    exit 1
  }
  [[ -f "${PROJECT_ROOT}/config/nav2_params.yaml" ]] || {
    echo "ERROR: incomplete project: ${PROJECT_ROOT}/config/nav2_params.yaml" >&2
    exit 1
  }
  [[ -f "${PROJECT_ROOT}/src/supermarket_sorting_nav2/navigation/manual_goal_bridge.py" ]] || {
    echo "ERROR: incomplete project: manual_goal_bridge.py is missing" >&2
    exit 1
  }
  [[ -f "${PROJECT_ROOT}/src/supermarket_sorting_nav2/navigation/scan_posture_initializer.py" ]] || {
    echo "ERROR: incomplete project: scan_posture_initializer.py is missing" >&2
    exit 1
  }
  [[ -f "${PROJECT_ROOT}/config/aruco_truth.json" ]] || {
    echo "ERROR: incomplete project: config/aruco_truth.json is missing" >&2
    exit 1
  }
  if [[ "$RUN_MODE" == "e_kele_cycle" || "$RUN_MODE" == "e_maidong_cycle" \
      || "$RUN_MODE" == "e_mixed_cycle" || "$RUN_MODE" == "random_cycle" ]]; then
    [[ -f "${PROJECT_ROOT}/src/supermarket_sorting_nav2/autonomous_sorting_mission.py" ]] || {
      echo "ERROR: incomplete project: autonomous_sorting_mission.py is missing" >&2
      exit 1
    }
  fi
  if [[ "$RUN_MODE" == "scan_only" || "$RUN_MODE" == "e_kele_cycle" \
      || "$RUN_MODE" == "e_maidong_cycle" \
      || "$RUN_MODE" == "e_mixed_cycle" || "$RUN_MODE" == "random_cycle" ]] \
      && [[ ! -f "$PRODUCT_WEIGHT_FILE" ]]; then
    echo "ERROR: nine-class weights not found: $PRODUCT_WEIGHT_FILE" >&2
    exit 1
  fi
  docker image inspect "$SERVER_IMAGE" >/dev/null 2>&1 || {
    echo "ERROR: Docker image not found: $SERVER_IMAGE" >&2
    exit 1
  }
  docker image inspect "$CLIENT_IMAGE" >/dev/null 2>&1 || {
    echo "ERROR: Docker image not found: $CLIENT_IMAGE" >&2
    exit 1
  }
  docker run --rm --entrypoint bash "$SERVER_IMAGE" -lc '
    server=/workspace/supermarket_sorting_task/examples/supermarket_sorting/supermarket_sorting_server.py
    grep -q "SUPERMARKET_RANDOMIZE" "$server" &&
    grep -q "SUPERMARKET_SEED" "$server" &&
    grep -q "SUPERMARKET_OBSTACLE_SEED" "$server" &&
    grep -q "SUPERMARKET_TASKS" "$server"
  ' || {
    echo "ERROR: Server image does not expose the required competition variables" >&2
    exit 1
  }
}


print_configuration() {
  cat <<EOF
===== supermarket_sorting_final competition test =====
project:                  $PROJECT_ROOT
server image:             $SERVER_IMAGE
client image:             $CLIENT_IMAGE
run mode:                 $RUN_MODE
controller:               $CONTROLLER
product randomization:    1
E three-kele debug:       $SUPERMARKET_E_THREE_KELE_DEBUG
E L3-right kele debug:    $SUPERMARKET_E_L3_RIGHT_KELE_DEBUG
E five-kele debug:        $SUPERMARKET_E_FIVE_KELE_DEBUG
E three-maidong debug:    $SUPERMARKET_E_THREE_MAIDONG_DEBUG
E mixed debug:            $SUPERMARKET_E_MIXED_DEBUG
E middle-food debug:      $SUPERMARKET_E_MIDDLE_FOOD_DEBUG
E middle gum/fruit debug: $SUPERMARKET_E_MIDDLE_GUM_FRUIT_DEBUG
E five-kind debug:        $SUPERMARKET_E_FIVE_KIND_DEBUG
E regression layout 9:    $SUPERMARKET_E_MIDDLE_TISSUE_DEBUG
E regression layout 10:   $SUPERMARKET_E_TISSUE_FIVE_KIND_DEBUG
E mixed sequence:         $SUPERMARKET_E_MIXED_SEQUENCE
forced first target:      ${SUPERMARKET_FORCE_FIRST_TARGET_KIND:-none}
product seed:             $PRODUCT_SEED
task seed:                $TASK_SEED
random obstacles:         $ENABLE_RANDOM_OBSTACLES
obstacle seed:            $OBSTACLE_SEED
five source task IDs:     $TASKS
nine-class weights:       $PRODUCT_WEIGHT_FILE
YOLO backend:             $PRODUCT_WEIGHT_BACKEND
inventory minimum votes:  $SUPERMARKET_INVENTORY_MIN_VOTES
scan stand-off:           $SUPERMARKET_SCAN_STAND_OFF m
approach stand-off:       $SUPERMARKET_APPROACH_STAND_OFF m
startup scan slide:       $SUPERMARKET_SCAN_SLIDE m
ArUco map source:         $PROJECT_ROOT/config/aruco_truth.json
log directory:            $LOG_DIR
EOF
}


log_event() {
  local role="$1"
  local event="$2"
  mkdir -p "$LOG_DIR"
  printf '%s role=%s event=%s\n' "$(date --iso-8601=seconds)" "$role" "$event" \
    | tee -a "${LOG_DIR}/timeline.log"
}


write_metadata() {
  mkdir -p "$LOG_DIR"
  {
    printf 'STARTED_AT=%q\n' "$(date --iso-8601=seconds)"
    printf 'RUN_ID=%q\n' "$RUN_ID"
    printf 'PROJECT_ROOT=%q\n' "$PROJECT_ROOT"
    printf 'RUN_MODE=%q\n' "$RUN_MODE"
    printf 'CONTROLLER=%q\n' "$CONTROLLER"
    printf 'PRODUCT_SEED=%q\n' "$PRODUCT_SEED"
    printf 'TASK_SEED=%q\n' "$TASK_SEED"
    printf 'OBSTACLE_SEED=%q\n' "$OBSTACLE_SEED"
    printf 'ENABLE_RANDOM_OBSTACLES=%q\n' "$ENABLE_RANDOM_OBSTACLES"
    printf 'SUPERMARKET_E_THREE_KELE_DEBUG=%q\n' "$SUPERMARKET_E_THREE_KELE_DEBUG"
    printf 'SUPERMARKET_E_L3_RIGHT_KELE_DEBUG=%q\n' "$SUPERMARKET_E_L3_RIGHT_KELE_DEBUG"
    printf 'SUPERMARKET_E_FIVE_KELE_DEBUG=%q\n' "$SUPERMARKET_E_FIVE_KELE_DEBUG"
    printf 'SUPERMARKET_E_THREE_MAIDONG_DEBUG=%q\n' "$SUPERMARKET_E_THREE_MAIDONG_DEBUG"
    printf 'SUPERMARKET_E_MIXED_DEBUG=%q\n' "$SUPERMARKET_E_MIXED_DEBUG"
    printf 'SUPERMARKET_E_MIDDLE_FOOD_DEBUG=%q\n' "$SUPERMARKET_E_MIDDLE_FOOD_DEBUG"
    printf 'SUPERMARKET_E_MIDDLE_GUM_FRUIT_DEBUG=%q\n' "$SUPERMARKET_E_MIDDLE_GUM_FRUIT_DEBUG"
    printf 'SUPERMARKET_E_FIVE_KIND_DEBUG=%q\n' "$SUPERMARKET_E_FIVE_KIND_DEBUG"
    printf 'SUPERMARKET_E_MIDDLE_TISSUE_DEBUG=%q\n' "$SUPERMARKET_E_MIDDLE_TISSUE_DEBUG"
    printf 'SUPERMARKET_E_TISSUE_FIVE_KIND_DEBUG=%q\n' "$SUPERMARKET_E_TISSUE_FIVE_KIND_DEBUG"
    printf 'SUPERMARKET_E_MIXED_SEQUENCE=%q\n' "$SUPERMARKET_E_MIXED_SEQUENCE"
    printf 'SUPERMARKET_FORCE_FIRST_TARGET_KIND=%q\n' "$SUPERMARKET_FORCE_FIRST_TARGET_KIND"
    printf 'TASKS=%q\n' "$TASKS"
    printf 'SUPERMARKET_INVENTORY_MIN_VOTES=%q\n' "$SUPERMARKET_INVENTORY_MIN_VOTES"
    printf 'SUPERMARKET_SCAN_STAND_OFF=%q\n' "$SUPERMARKET_SCAN_STAND_OFF"
    printf 'SUPERMARKET_APPROACH_STAND_OFF=%q\n' "$SUPERMARKET_APPROACH_STAND_OFF"
    printf 'SUPERMARKET_SCAN_SLIDE=%q\n' "$SUPERMARKET_SCAN_SLIDE"
    printf 'ARUCO_TRUTH_FILE=%q\n' "${PROJECT_ROOT}/config/aruco_truth.json"
    printf 'SERVER_IMAGE=%q\n' "$SERVER_IMAGE"
    printf 'CLIENT_IMAGE=%q\n' "$CLIENT_IMAGE"
    if [[ -f "$PRODUCT_WEIGHT_FILE" ]]; then
      printf 'PRODUCT_WEIGHT_FILE=%q\n' "$PRODUCT_WEIGHT_FILE"
      printf 'PRODUCT_WEIGHT_SHA256=%q\n' "$(sha256sum "$PRODUCT_WEIGHT_FILE" | awk '{print $1}')"
    fi
    printf 'PRODUCT_WEIGHT_BACKEND=%q\n' "$PRODUCT_WEIGHT_BACKEND"
    if [[ -f "$PRODUCT_PT_WEIGHT_FILE" ]]; then
      printf 'PRODUCT_PT_WEIGHT_SHA256=%q\n' "$(sha256sum "$PRODUCT_PT_WEIGHT_FILE" | awk '{print $1}')"
    fi
    printf 'SERVER_IMAGE_ID=%q\n' "$(docker image inspect "$SERVER_IMAGE" --format '{{.Id}}')"
    printf 'CLIENT_IMAGE_ID=%q\n' "$(docker image inspect "$CLIENT_IMAGE" --format '{{.Id}}')"
  } > "${LOG_DIR}/metadata.env"
}


run_server() {
  log_event server start
  set +e
  docker run --rm -it \
    --pull=never \
    --gpus all \
    --network host \
    --ipc host \
    --name "$SERVER_CONTAINER" \
    -e "DISPLAY=${DISPLAY}" \
    -e ROS_DOMAIN_ID=99 \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e MUJOCO_GL=glfw \
    -e SUPERMARKET_HEADLESS=0 \
    -e SUPERMARKET_ENABLE_RENDER=1 \
    -e SUPERMARKET_ENABLE_LIDAR=1 \
    -e SUPERMARKET_USE_GS=1 \
    -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
    -e SUPERMARKET_FIXED_BASELINE=0 \
    -e SUPERMARKET_RANDOMIZE=1 \
    -e "SUPERMARKET_SEED=${PRODUCT_SEED}" \
    -e "SUPERMARKET_RANDOMIZE_OBSTACLES=${ENABLE_RANDOM_OBSTACLES}" \
    -e "SUPERMARKET_OBSTACLE_SEED=${OBSTACLE_SEED}" \
    -e "SUPERMARKET_TASKS=${TASKS}" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "${CACHE_VOLUME}:/root/.cache" \
    "$SERVER_IMAGE" \
    bash -lc 'cd /workspace/supermarket_sorting_task && source /opt/ros/humble/setup.bash && exec python3 examples/supermarket_sorting/supermarket_sorting_server.py' \
    2>&1 | tee "${LOG_DIR}/server.log"
  local status=${PIPESTATUS[0]}
  set -e
  log_event server "exit_${status}"
  return "$status"
}


wait_for_container() {
  # 每秒检查一次容器状态；timeout_sec 是总等待上限，不是固定停顿。
  local container="$1"
  local timeout_sec="$2"
  local deadline=$((SECONDS + timeout_sec))
  while (( SECONDS < deadline )); do
    if [[ "$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)" == "true" ]]; then
      return 0
    fi
    sleep 1
  done
  echo "ERROR: container did not become ready: $container" >&2
  return 1
}


capture_task_message() {
  log_event client capture_task_start
  # 最多等 30 s 获取 Server 的一次性任务消息；失败只告警，不中止 Client。
  if timeout 30 docker exec "$CLIENT_CONTAINER" bash -lc \
    'source /opt/ros/humble/setup.bash && ros2 topic echo /supermarket_sorting/task std_msgs/msg/String --once --field data --full-length --qos-reliability reliable --qos-durability transient_local' \
    2>&1 | tee "${LOG_DIR}/task_message.txt"; then
    log_event client capture_task_complete
  else
    log_event client capture_task_timeout
    echo "WARNING: task message was not captured within 30 seconds" >&2
  fi
}


run_client() {
  log_event client wait_server
  wait_for_container "$SERVER_CONTAINER" 60  # Server 容器启动上限 60 s

  # 给仿真器 6 s 完成模型、传感器和 ROS 发布器初始化；随后还会单独检查 Nav2。
  sleep 6

  log_event client container_start
  docker run --rm -dit \
    --pull=never \
    --gpus all \
    --network host \
    --ipc host \
    --name "$CLIENT_CONTAINER" \
    -e "DISPLAY=${DISPLAY}" \
    -e ROS_DOMAIN_ID=99 \
    -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    -e "RUN_MODE=${RUN_MODE}" \
    -e "RUN_ID=${RUN_ID}" \
    -e "CONTROLLER=${CONTROLLER}" \
    -e "FULL_ARM_CLEARANCE_VERIFIED=${FULL_ARM_CLEARANCE_VERIFIED}" \
    -e "SUPERMARKET_INVENTORY_MIN_VOTES=${SUPERMARKET_INVENTORY_MIN_VOTES}" \
    -e "SUPERMARKET_SCAN_STAND_OFF=${SUPERMARKET_SCAN_STAND_OFF}" \
    -e "SUPERMARKET_APPROACH_STAND_OFF=${SUPERMARKET_APPROACH_STAND_OFF}" \
    -e "SUPERMARKET_SCAN_SLIDE=${SUPERMARKET_SCAN_SLIDE}" \
    -e "SUPERMARKET_E_MIXED_SEQUENCE=${SUPERMARKET_E_MIXED_SEQUENCE}" \
    -e "SUPERMARKET_FORCE_FIRST_TARGET_KIND=${SUPERMARKET_FORCE_FIRST_TARGET_KIND}" \
    -e "SUPERMARKET_PRODUCT_WEIGHTS=/workspace/baseline/weights/$(basename "$PRODUCT_WEIGHT_FILE")" \
    -e "SUPERMARKET_DETECTOR_DEVICE=cuda" \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    -v "${PROJECT_ROOT}:/workspace/baseline:rw" \
    "$CLIENT_IMAGE" \
    bash >/dev/null

  cleanup_client() {
    docker stop "$CLIENT_CONTAINER" >/dev/null 2>&1 || true
  }
  trap cleanup_client EXIT INT TERM

  capture_task_message
  log_event client mission_start
  set +e
  docker exec -it \
    -e "RUN_MODE=${RUN_MODE}" \
    -e "CONTROLLER=${CONTROLLER}" \
    -e "FULL_ARM_CLEARANCE_VERIFIED=${FULL_ARM_CLEARANCE_VERIFIED}" \
    "$CLIENT_CONTAINER" \
    bash -lc 'source /opt/ros/humble/setup.bash && cd /workspace/baseline && exec ./scripts/run_nav2_mission.sh "$RUN_MODE"' \
    2>&1 | tee "${LOG_DIR}/client.log"
  local status=${PIPESTATUS[0]}
  set -e
  log_event client "mission_exit_${status}"
  return "$status"
}


run_rviz() {
  log_event rviz wait_client
  wait_for_container "$CLIENT_CONTAINER" 120  # RViz 最多等 Client 容器 120 s
  log_event rviz start
  set +e
  docker exec -it "$CLIENT_CONTAINER" \
    bash -lc 'source /opt/ros/humble/setup.bash && exec rviz2 -d /workspace/baseline/rviz/supermarket_nav2.rviz' \
    2>&1 | tee "${LOG_DIR}/rviz.log"
  local status=${PIPESTATUS[0]}
  set -e
  log_event rviz "exit_${status}"
  return "$status"
}


stop_final_containers() {
  docker rm -f "$SERVER_CONTAINER" "$CLIENT_CONTAINER" >/dev/null 2>&1 || true
  echo "Stopped $SERVER_CONTAINER and $CLIENT_CONTAINER (if present)."
}


case "$ROLE" in
  -h|--help)
    usage
    exit 0
    ;;
  --stop)
    stop_final_containers
    exit 0
    ;;
esac

validate_options

case "$ROLE" in
  __server) run_server; exit ;;
  __client) run_client; exit ;;
  __rviz) run_rviz; exit ;;
  --check)
    verify_host
    print_configuration
    echo "CHECK_OK: launcher, images, options and Server interfaces are valid."
    exit 0
    ;;
  "") ;;
  *) echo "ERROR: unknown argument: $ROLE" >&2; usage; exit 2 ;;
esac

verify_host
command -v gnome-terminal >/dev/null || {
  echo "ERROR: gnome-terminal is unavailable" >&2
  exit 1
}
command -v xhost >/dev/null || { echo "ERROR: xhost is unavailable" >&2; exit 1; }
: "${DISPLAY:?ERROR: DISPLAY is not set; Docker GUI cannot be opened}"

# Do not silently kill a historical Nav2 test that may still be running.
for incompatible in \
  supermarket_sorting_server supermarket_sorting_client \
  supermarket_sorting_full_server supermarket_sorting_full_client; do
  if [[ "$(docker inspect -f '{{.State.Running}}' "$incompatible" 2>/dev/null || true)" == "true" ]]; then
    echo "ERROR: old test container is running: $incompatible" >&2
    echo "Stop the old run before starting supermarket_sorting_final." >&2
    exit 1
  fi
done

# Remove only stale containers owned by this launcher.
docker rm -f "$SERVER_CONTAINER" "$CLIENT_CONTAINER" >/dev/null 2>&1 || true
docker volume create "$CACHE_VOLUME" >/dev/null
xhost +local:docker >/dev/null 2>&1

write_metadata
log_event launcher start
print_configuration | tee "${LOG_DIR}/launch_config.txt"

export PROJECT_ROOT SERVER_IMAGE CLIENT_IMAGE SERVER_CONTAINER CLIENT_CONTAINER
export PRODUCT_WEIGHT_FILE PRODUCT_PT_WEIGHT_FILE
export PRODUCT_WEIGHT_BACKEND
export CACHE_VOLUME RUN_MODE CONTROLLER ENABLE_RANDOM_OBSTACLES
export FULL_ARM_CLEARANCE_VERIFIED TASKS TASK_SEED PRODUCT_SEED OBSTACLE_SEED
export SUPERMARKET_E_THREE_KELE_DEBUG
export SUPERMARKET_E_L3_RIGHT_KELE_DEBUG
export SUPERMARKET_E_FIVE_KELE_DEBUG
export SUPERMARKET_E_THREE_MAIDONG_DEBUG
export SUPERMARKET_E_MIXED_DEBUG
export SUPERMARKET_E_MIDDLE_FOOD_DEBUG SUPERMARKET_E_MIDDLE_GUM_FRUIT_DEBUG
export SUPERMARKET_E_FIVE_KIND_DEBUG
export SUPERMARKET_E_MIDDLE_TISSUE_DEBUG SUPERMARKET_E_TISSUE_FIVE_KIND_DEBUG
export SUPERMARKET_E_MIXED_SEQUENCE
export SUPERMARKET_FORCE_FIRST_TARGET_KIND
export SUPERMARKET_INVENTORY_MIN_VOTES SUPERMARKET_SCAN_STAND_OFF
export SUPERMARKET_APPROACH_STAND_OFF SUPERMARKET_SCAN_SLIDE
export RUN_ID LOG_BASE LOG_DIR DISPLAY

printf -v server_command '%q %q' "$SELF" __server
printf -v client_command '%q %q' "$SELF" __client
printf -v rviz_command '%q %q' "$SELF" __rviz

gnome-terminal --window \
  --title="Final Competition Server" --command="$server_command" \
  --tab --title="Final Nav2 Client" --command="$client_command" \
  --tab --title="Final RViz" --command="$rviz_command"

echo "Started competition test. Logs: $LOG_DIR"
if [[ "$RUN_MODE" == "e_kele_cycle" ]]; then
  cat <<'EOF'
After the Client prints E_KELE_CYCLE_READY, start the mission from another terminal:
  docker exec supermarket_sorting_final_client bash -lc 'source /opt/ros/humble/setup.bash; ros2 topic pub -r 1 /supermarket_sorting/mission_command std_msgs/msg/String "data: clear_e_kele"'
Status: /supermarket_sorting/mission_status
EOF
fi
if [[ "$RUN_MODE" == "e_maidong_cycle" ]]; then
  cat <<'EOF'
After the Client prints E_MAIDONG_CYCLE_READY, start the mission from another terminal:
  docker exec supermarket_sorting_final_client bash -lc 'source /opt/ros/humble/setup.bash; ros2 topic pub -r 1 /supermarket_sorting/mission_command std_msgs/msg/String "data: clear_e_maidong"'
Status: /supermarket_sorting/mission_status
EOF
fi
if [[ "$RUN_MODE" == "e_mixed_cycle" ]]; then
  cat <<EOF
After the Client prints E_MIXED_CYCLE_READY, start the ordered mixed mission:
  docker exec supermarket_sorting_final_client bash -lc 'source /opt/ros/humble/setup.bash; ros2 topic pub -r 1 /supermarket_sorting/mission_command std_msgs/msg/String "data: clear_e_mixed"'
Order: ${SUPERMARKET_E_MIXED_SEQUENCE//,/ -> }
Status: /supermarket_sorting/mission_status
EOF
fi
if [[ "$RUN_MODE" == "random_cycle" ]]; then
  cat <<EOF
After the Client prints RANDOM_CYCLE_READY, start the autonomous multi-shelf mission:
  docker exec supermarket_sorting_final_client bash -lc 'source /opt/ros/humble/setup.bash; ros2 topic pub -r 1 /supermarket_sorting/mission_command std_msgs/msg/String "data: clear_random"'
Random source bodies: $TASKS
Seeds: task=$TASK_SEED products=$PRODUCT_SEED obstacles=$OBSTACLE_SEED
Status: /supermarket_sorting/mission_status
EOF
fi
