"""标定文件的加载与校验。

由 lightfield_bridge/lightfield_litenet/lfnav/ros_calibration.py 提取，逻辑等价：
ROS 侧用的是「所选 3x3 里相邻相机的实际物理间距」，不是 HCI 数据集的单位间距。
"""

import json

import numpy as np

# 必须由使用者填实测值的字段
REQUIRED_KEYS = (
    "width",
    "height",
    "fx",
    "fy",
    "cx",
    "cy",
    "selected_baseline_m",
    "disparity_offset_px",
    "disparity_sign",
    "angular_step",
)


def load_calibration(path):
    """读标定 JSON，校验后补上 baseline_m = selected_baseline_m / angular_step。"""
    with open(path, encoding="utf-8") as handle:
        c = json.load(handle)

    if any(c.get(k) is None or not np.isfinite(float(c[k])) for k in REQUIRED_KEYS):
        raise ValueError("标定文件字段不全或存在非有限值，请填写实测值")

    if c["width"] != 512 or c["height"] != 512 or c["angular_step"] != 2:
        raise ValueError("本 ONNX 要求单视角 512x512 且 angular_step=2")

    if min(c["fx"], c["fy"], c["selected_baseline_m"]) <= 0 or c["disparity_sign"] not in (-1, 1):
        raise ValueError("fx/fy/selected_baseline_m 必须为正，disparity_sign 只能是 ±1")

    if not isinstance(c.get("frame_id"), str) or not c["frame_id"]:
        raise ValueError("frame_id 必须给出中心相机光学坐标系的名字")

    # 网络的视差以其训练时的相邻视角步长为单位，所以物理基线要按 angular_step 折算
    c["baseline_m"] = c["selected_baseline_m"] / c["angular_step"]
    return c
