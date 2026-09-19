"""视差 → 米制深度 → 点云。

由 lightfield_bridge/lightfield_litenet/lfnav/geometry.py 提取，
只保留推理/部署需要的三个函数；读 PFM、HCI 标定、训练评估那些函数留在了原仓库。

公式与原来完全一致：
    Z = fx * B / (disparity_sign * d + disparity_offset_px)
其中 B = selected_baseline_m / angular_step 已在 calibration.load_calibration 里算好。
"""

import numpy as np


def disparity_to_depth(disp, c):
    """视差图 → 米制 Z 深度；分母非正处留 NaN。"""
    denom = c["disparity_sign"] * np.asarray(disp) + c["disparity_offset_px"]
    z = np.full(denom.shape, np.nan, dtype=np.float32)
    np.divide(c["fx"] * c["baseline_m"], denom, out=z,
              where=np.isfinite(denom) & (denom > 1e-6))
    return z


def navigation_outputs(disp, log_scale, c, min_depth=0.15, max_depth=10.0,
                       max_scale=0.15, max_relative_depth_error=0.1, border=11):
    """网络两通道输出 → (depth, valid, depth_scale)。

    log_scale 是 Laplace 视差误差尺度（取 exp 后是像素单位），不是置信概率。
    border 像素的边缘会被判为无效：那里没有完整的邻域支撑。
    """
    if disp.shape != (c["height"], c["width"]):
        raise ValueError("分辨率与标定不符；要改分辨率必须同时换算内参与 offset")
    if min_depth <= 0 or max_depth <= min_depth or max_scale <= 0 or max_relative_depth_error <= 0:
        raise ValueError("导航距离/不确定度阈值不合法")
    if log_scale.shape != disp.shape or border < 0 or 2 * border >= min(disp.shape):
        raise ValueError("不确定度图形状或 border 不合法")

    z = disparity_to_depth(disp, c)
    b = np.exp(np.clip(log_scale, -5, 3))
    denom = c["disparity_sign"] * disp + c["disparity_offset_px"]
    relative_error = b / np.maximum(denom, 1e-6)
    valid = (np.isfinite(z) & np.isfinite(log_scale) & (z >= min_depth) &
             (z <= max_depth) & (b <= max_scale) & (relative_error <= max_relative_depth_error))
    if border:
        valid[:border] = valid[-border:] = False
        valid[:, :border] = valid[:, -border:] = False

    depth = np.where(valid, z, np.nan).astype(np.float32)
    # 一阶传播得到的 Laplace 误差尺度，不是标定过的置信概率
    depth_scale = z * z / (c["fx"] * c["baseline_m"]) * b
    return depth, valid, depth_scale.astype(np.float32)


def point_cloud(depth, c, stride=2):
    """深度图 → (N, 3) float32 点云，中心相机光学系（x 右、y 下、z 前）。"""
    if stride < 1:
        raise ValueError("stride 必须 >= 1")
    y, x = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[::stride, ::stride]
    xyz = np.stack(((x - c["cx"]) * z / c["fx"], (y - c["cy"]) * z / c["fy"], z), axis=-1)
    return xyz[np.isfinite(z) & (z > 0)].astype(np.float32)
