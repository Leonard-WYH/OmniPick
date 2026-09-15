"""控制循环辅助函数。

``step_func`` 每个控制周期只允许当前位置向目标移动给定步长，用于限制
机械臂、升降轴和夹爪命令的突变；它本身不产生 sleep 或固定等待。
"""


def step_func(current, target, step):
    """以不超过 ``step`` 的增量逼近目标，并避免跨过目标值。"""
    if current < target - step:
        return current + step
    if current > target + step:
        return current - step
    return target
