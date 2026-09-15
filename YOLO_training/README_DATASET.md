# YOLO 九类商品数据集生成

本目录通过 `supermarket_sorting:server` 镜像提供完整的 Discoverse、MuJoCo、
3DGS 场景和商品资源，只把本目录中的 `perception` 挂载进镜像，因此不会修改
Nav2、Baseline 或 Docker 镜像。

## 默认生成

```bash
cd ~/OmniPick/YOLO_training
./generate_dataset.sh
```

默认生成 1600 个不同相机/布局视角，每个视角保存原图和 1 张外观增强图，
合计 3200 张图片。输出位于：

```text
my_baseline/examples/supermarket_sorting/perception/dataset/
```

重点检查：

```text
dataset/debug/contact_sheet_*.jpg   # 带实例掩码和紧框的抽样总览
dataset/debug/frame_*.jpg           # 单帧标注预览
dataset/stats.json                  # 类别、训练集和验证集统计
dataset/manifest.jsonl              # 每个基础视角的相机姿态和框质量数据
```

## 先生成小规模预览

```bash
./generate_dataset.sh \
  --frames 80 \
  --variants 0 \
  --pose-mode diverse \
  --out /workspace/supermarket_sorting_task/examples/supermarket_sorting/perception/dataset_preview \
  --debug-overlay \
  --debug-max 80 \
  --overwrite
```

## 标注原理

生成器为 45 个商品实例分别分配高维特征码，并额外进行一次保持真实遮挡关系的
GS 特征渲染。每个像素解码为可见商品实例，再从有效实例掩码的主要连通区域
生成紧框。过小、过碎、掩码填充率过低的框以及有效商品少于 4 个的视角会被
丢弃。最终框不再依赖手填的近似三维尺寸。

## 测试训练后的权重

训练结束并生成 `perception/checkpoints/products.pt` 后，运行：

```bash
cd ~/OmniPick/YOLO_training
./test_yolo.sh
```

默认会对完整 `val` 集计算 9 类商品的 Precision、Recall、mAP50 和
mAP50-95，并随机抽取 48 张未增强的 `v0` 验证图片生成“真实标注/模型预测”
左右对照图。结果保存在：

```text
my_baseline/examples/supermarket_sorting/perception/runs/products_9class_test/
├── summary.json                 # 总指标、逐类指标和抽样预测详情
├── per_class_metrics.csv        # 9 类商品的独立指标
├── validation/                  # Ultralytics 混淆矩阵和 PR/F1 曲线
├── comparisons/                 # 每张图片的 GT/预测对照图
└── contact_sheet_*.jpg          # 对照图联系表，适合快速目检
```

只快速查看 24 张图片、不重新跑完整验证集：

```bash
./test_yolo.sh --skip-val --samples 24
```

调整可视化置信度阈值（完整验证指标仍按 Ultralytics 的验证阈值计算）：

```bash
./test_yolo.sh --conf 0.40 --samples 48
```

## 注意

`--overwrite` 会删除 `--out` 指定的旧输出目录。需要保留旧数据时，请使用新的
`--out` 路径或先备份原目录。
