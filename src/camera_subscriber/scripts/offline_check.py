#!/usr/bin/python3
"""离线自检：合成已知深度的平面光场 -> ONNX -> 视差/深度。

不用开 Gazebo 就能回答三个问题：

1. onnxruntime 能不能加载权重、输入形状对不对；
2. 拼图的行列与图像水平/竖直方向的对应关系对不对（列方向 = 水平子孔径位移）；
3. 标定里的 ``disparity_sign`` 与 ``selected_baseline_m`` 换算对不对 —— 看
   "Z_est" 列是否等于 "Z_true" 列。

用法（用系统解释器，别用 conda/venv 里的）：

    /usr/bin/python3 scripts/offline_check.py                    # 默认 0.5/1/2/4 m
    /usr/bin/python3 scripts/offline_check.py -z 0.5 1.0         # 指定深度
    /usr/bin/python3 scripts/offline_check.py -t 纹理.png        # 换纹理（默认随机噪声）
    /usr/bin/python3 scripts/offline_check.py -m h               # 只保留水平子孔径位移

``-m h`` 只有列方向位移、``-m v`` 只有行方向位移：单看某一个方向时模型的表现会退化
（训练数据里两个方向都有视差），判别方向要看哪一个能给出接近 Z_true 的 Z_est。
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera_subscriber.calibration import load_calibration  # noqa: E402
from camera_subscriber.geometry import navigation_outputs  # noqa: E402
from camera_subscriber.inference import OnnxRunner, build_mosaic, mosaic_to_input  # noqa: E402

PAD = 96
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def texture_from(path, c, seed=0):
    """返回 (H+2*PAD, W+2*PAD, 3) float [0,1]，用作平面贴图。"""
    rng = np.random.default_rng(seed)
    need_h, need_w = c["height"] + 2 * PAD, c["width"] + 2 * PAD
    if path:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit("读不到纹理: %s" % path)
        reps = (int(np.ceil(need_h / image.shape[0])),
                int(np.ceil(need_w / image.shape[1])), 1)
        return np.tile(image, reps)[:need_h, :need_w].astype(np.float32) / 255.0
    gray = np.clip(rng.normal(0.5, 0.25, (need_h, need_w)), 0, 1).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (0, 0), 1.6)
    return np.repeat(np.clip(gray, 0, 1)[..., None], 3, axis=2)


def make_views(c, texture, z_true, mode="both"):
    """按 URDF 角坐标生成 9 视角：index//3 -> 行(竖直 u)，index%3 -> 列(水平 v)。

    u 沿 stand +X = base -Z（向下），v 沿 stand +Y = base +Y（向左），所以：
    相机沿 base +Y 平移 +Δ，像面内容沿 +x 平移 Δ*fx/Z；
    相机沿 base -Z（向下）平移 +Δ，像面内容沿 -y 平移 Δ*fy/Z。
    """
    h, w = c["height"], c["width"]
    step = c["selected_baseline_m"] / c["angular_step"]      # dense step (m)
    frames = []
    for index in range(9):
        u = (index // 3 - 1) * c["angular_step"]
        v = (index % 3 - 1) * c["angular_step"]
        dx = v * step * c["fx"] / z_true if mode in ("both", "h") else 0.0
        dy = -u * step * c["fy"] / z_true if mode in ("both", "v") else 0.0
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        big = cv2.warpAffine(texture, matrix, (texture.shape[1], texture.shape[0]),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        frames.append((np.clip(big[PAD:PAD + h, PAD:PAD + w], 0, 1) * 255).astype(np.uint8))
    return frames


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--calibration",
                        default=os.path.join(HERE, "config", "lf_camera_sim.json"))
    parser.add_argument("-p", "--model", default=os.path.join(HERE, "models", "best.onnx"))
    parser.add_argument("-t", "--texture", default="", help="平面贴图（默认随机噪声）")
    parser.add_argument("-z", "--depths", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    parser.add_argument("-m", "--mode", choices=("both", "h", "v"), default="both",
                        help="h=只留水平子孔径位移, v=只留竖直, both=两者都有（真实情况）")
    parser.add_argument("--provider", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--dump", default="/tmp/lf_offline_mosaic.png",
                        help="拼图落盘路径，用来看视角顺序；空字符串则不落盘")
    args = parser.parse_args(argv)

    c = load_calibration(args.calibration)
    runner = OnnxRunner(args.model, args.provider)
    texture = texture_from(args.texture, c)
    print("标定 %s | 模型 %s" % (args.calibration, args.model))
    print("输入 %s | %s | 纹理 %s | 位移模式 %s"
          % (runner.shape, runner.providers, args.texture or "随机噪声", args.mode))

    print("%8s %10s %10s %7s %9s %8s %8s"
          % ("Z_true", "d_true_px", "d_pred_px", "比值", "Z_est", "valid%", "耗时s"))
    for z_true in args.depths:
        mosaic = build_mosaic(make_views(c, texture, z_true, args.mode),
                              c["height"], c["width"])
        if args.dump:
            cv2.imwrite(args.dump, mosaic[..., ::-1])
        x = mosaic_to_input(mosaic, c["height"], c["width"])
        started = time.perf_counter()
        prediction = runner(x)[0]
        elapsed = time.perf_counter() - started
        if not np.isfinite(prediction).all():
            print("%8.2f  模型输出存在非有限值，权重或输入有问题" % z_true)
            continue
        disp, log_scale = prediction[0], prediction[1]
        core = (slice(c["height"] // 4, 3 * c["height"] // 4),
                slice(c["width"] // 4, 3 * c["width"] // 4))
        d_pred = float(np.median(disp[core]))
        d_true = c["fx"] * c["baseline_m"] / z_true
        depth, valid, _ = navigation_outputs(disp, log_scale, c, border=11)
        finite = depth[np.isfinite(depth)]
        z_est = float(np.median(finite)) if finite.size else float("nan")
        print("%8.2f %10.3f %10.3f %7.2f %9.3f %7.1f%% %8.2f"
              % (z_true, d_true, d_pred, d_pred / d_true, z_est,
                 100 * valid.mean(), elapsed))
    if args.dump:
        print("拼图已写到 %s（放最前面的是 index 0 视角，行优先）" % args.dump)
    return 0


if __name__ == "__main__":
    sys.exit(main())
