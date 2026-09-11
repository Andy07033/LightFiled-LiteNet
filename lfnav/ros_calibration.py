"""ROS calibration uses physical spacing between selected views, not HCI unit spacing."""
import json
import numpy as np

def load_ros_calibration(path):
    with open(path, encoding='utf-8') as f:
        c = json.load(f)
    keys = ('width', 'height', 'fx', 'fy', 'cx', 'cy', 'selected_baseline_m',
            'disparity_offset_px', 'disparity_sign', 'angular_step')
    if any(c.get(k) is None or not np.isfinite(float(c[k])) for k in keys):
        raise ValueError('Fill all calibration fields with measured values')
    if c['width'] != 512 or c['height'] != 512 or c['angular_step'] != 2:
        raise ValueError('Supplied ONNX requires 512x512 per view and step=2 convention')
    if min(c['fx'], c['fy'], c['selected_baseline_m']) <= 0 or c['disparity_sign'] not in (-1, 1):
        raise ValueError('Invalid focal length, baseline or disparity sign')
    if not isinstance(c.get('frame_id'), str) or not c['frame_id']:
        raise ValueError('frame_id must name the center optical frame')
    c['baseline_m'] = c['selected_baseline_m'] / c['angular_step']
    return c
