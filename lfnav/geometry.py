"""Signed disparity uses the original HCI angular unit, including step-2 inputs."""
import configparser
import json
from pathlib import Path

import numpy as np


def read_pfm(path):
    with open(path, "rb") as f:
        def line():
            s = f.readline()
            while s.startswith(b"#"):
                s = f.readline()
            return s.strip()
        magic = line()
        if magic not in (b"Pf", b"PF"):
            raise ValueError(f"Invalid PFM: {path}")
        w, h = map(int, line().split())
        scale = float(line())
        if scale == 0:
            raise ValueError("PFM scale cannot be zero")
        data = np.fromfile(f, dtype="<f4" if scale < 0 else ">f4")
        shape = (h, w) if magic == b"Pf" else (h, w, 3)
        return (np.flipud(data.reshape(shape)) * abs(scale)).astype(np.float32)


def write_pfm(path, data):
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("Expected a scalar disparity map")
    with open(path, "wb") as f:
        f.write(f"Pf\n{data.shape[1]} {data.shape[0]}\n-1\n".encode())
        np.flipud(data).astype("<f4").tofile(f)


def hci_calibration(path):
    c = configparser.ConfigParser()
    if not c.read(path):
        raise FileNotFoundError(path)
    i, e = c["intrinsics"], c["extrinsics"]
    w, h = int(i["image_resolution_x_px"]), int(i["image_resolution_y_px"])
    f = float(i["focal_length_mm"]) * max(w, h) / float(i["sensor_size_mm"])
    fb = f * float(e["baseline_mm"]) / 1000
    return dict(width=w, height=h, fx=f, fy=f, cx=(w-1)/2, cy=(h-1)/2,
                baseline_m=float(e["baseline_mm"])/1000,
                disparity_offset_px=fb/float(e["focus_distance_m"]),
                disparity_sign=1, angular_step=1, frame_id="lf_camera_optical_frame")


def load_calibration(path):
    c = hci_calibration(path) if Path(path).suffix == ".cfg" else json.loads(Path(path).read_text())
    required = ("width", "height", "fx", "fy", "cx", "cy", "baseline_m",
                "disparity_offset_px", "disparity_sign", "angular_step", "frame_id")
    if any(c.get(k) is None for k in required):
        raise ValueError("Calibration has missing/null fields. Supply measured camera calibration.")
    if not all(np.isfinite(float(c[k])) for k in required if k != "frame_id"):
        raise ValueError("Calibration contains nonfinite values")
    if min(c[k] for k in ("width", "height", "fx", "fy", "baseline_m")) <= 0:
        raise ValueError("Calibration dimensions, focal lengths and baseline must be positive")
    if c["angular_step"] != 1 or c["disparity_sign"] not in (-1, 1):
        raise ValueError("This model requires adjacent 3x3 views, and disparity_sign +/-1")
    return c


def disparity_to_depth(disp, c):
    denom = c["disparity_sign"] * np.asarray(disp) + c["disparity_offset_px"]
    z = np.full(denom.shape, np.nan, dtype=np.float32)
    np.divide(c["fx"] * c["baseline_m"], denom, out=z,
              where=np.isfinite(denom) & (denom > 1e-6))
    return z


def navigation_outputs(disp, log_scale, c, min_depth=0.15, max_depth=10.,
                       max_scale=0.15, max_relative_depth_error=0.1, border=11):
    if disp.shape != (c["height"], c["width"]):
        raise ValueError("Resolution differs from calibration; rescale intrinsics/offset explicitly")
    if min_depth <= 0 or max_depth <= min_depth or max_scale <= 0 or max_relative_depth_error <= 0:
        raise ValueError("Invalid navigation range/uncertainty thresholds")
    if log_scale.shape != disp.shape or border < 0 or 2*border >= min(disp.shape):
        raise ValueError("Invalid uncertainty shape or border")
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
    # First-order propagated Laplace error scale, not a calibrated confidence probability.
    depth_scale = z*z/(c["fx"]*c["baseline_m"]) * b
    return depth, valid, depth_scale.astype(np.float32)


def point_cloud(depth, c, stride=2):
    if stride < 1:
        raise ValueError("stride must be >= 1")
    y, x = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    z = depth[::stride, ::stride]
    xyz = np.stack(((x-c["cx"])*z/c["fx"], (y-c["cy"])*z/c["fy"], z), axis=-1)
    return xyz[np.isfinite(z) & (z > 0)].astype(np.float32)
