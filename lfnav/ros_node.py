"""ROS2 adapter; driver publishes synchronized rectified 3x3 RGB tile Image."""
from types import SimpleNamespace
from pathlib import Path
from ament_index_python.packages import get_package_share_directory
import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue

from .geometry import navigation_outputs, point_cloud
from .ros_calibration import load_ros_calibration
from .runtime import OnnxRunner, unpack_grid


class LFGeometryNode(Node):
    def __init__(self):
        super().__init__("lfnav_geometry")
        share = Path(get_package_share_directory('lightfield_litenet'))
        defaults = dict(model_path=str(share/'weights/best.onnx'), calibration='', provider='auto',
                        max_scale=0.15, max_depth=10.0, max_age=1.0,
                        min_valid_fraction=0.1, point_stride=2)
        args = SimpleNamespace(**{k: self.declare_parameter(k, v).value for k, v in defaults.items()})
        if not args.calibration:
            raise ValueError('Set calibration to a measured ROS calibration JSON file')
        if (not all(np.isfinite(v) for v in (args.max_age, args.max_depth, args.max_scale, args.min_valid_fraction))
                or min(args.max_age, args.max_scale) <= 0 or args.max_depth <= 0.15
                or not 0 < args.min_valid_fraction <= 1 or args.point_stride < 1):
            raise ValueError('Invalid geometry/freshness parameters')
        self.args, self.c = args, load_ros_calibration(args.calibration)
        self.runner = OnnxRunner(args.model_path, args.provider)
        self.get_logger().info('ONNX providers: ' + str(self.runner.providers))
        if self.runner.providers[0] == 'CPUExecutionProvider':
            self.get_logger().warning('Using CPU inference; verify processing rate before navigation')
        if self.runner.shape != (1, 27, self.c["height"], self.c["width"]):
            raise ValueError("Engine and calibration shape mismatch")
        self.qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                              history=HistoryPolicy.KEEP_LAST)
        self.disp_pub = self.create_publisher(Image, "/lf/disparity", self.qos)
        self.scale_pub = self.create_publisher(Image, "/lf/uncertainty_scale", self.qos)
        self.depth_pub = self.create_publisher(Image, "/lf/depth", self.qos)
        self.valid_pub = self.create_publisher(Image, "/lf/valid", self.qos)
        self.info_pub = self.create_publisher(CameraInfo, "/lf/camera_info", self.qos)
        from sensor_msgs.msg import PointCloud2
        self.cloud_pub = self.create_publisher(PointCloud2, "/lf/points", self.qos)
        self.diag_pub = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.last_success = None
        self.last_log = 0.
        self.create_subscription(Image, "/lf/image_grid", self.on_image, self.qos)
        self.create_timer(0.5, self.watchdog)

    def diagnostic(self, level, message, values=None):
        d = DiagnosticArray()
        d.header.stamp = self.get_clock().now().to_msg()
        status = DiagnosticStatus(level=level, name="lfnav/geometry", message=message,
                                  hardware_id=self.c["frame_id"])
        status.values = [KeyValue(key=k, value=str(v)) for k, v in (values or {}).items()]
        d.status = [status]
        self.diag_pub.publish(d)

    def watchdog(self):
        if self.last_success is None or (self.get_clock().now().nanoseconds-self.last_success)/1e9 > self.args.max_age or (self.last_success is not None and self.get_clock().now().nanoseconds < self.last_success):
            self.diagnostic(DiagnosticStatus.ERROR, "No fresh valid LF geometry; navigation supervisor must handle unknown space")

    def image(self, array, encoding, header):
        msg = Image()
        msg.header = header
        msg.height, msg.width = array.shape
        msg.encoding, msg.is_bigendian = encoding, False
        msg.step = msg.width*array.dtype.itemsize
        msg.data = np.ascontiguousarray(array).tobytes()
        return msg

    def on_image(self, msg):
        start = time.perf_counter()
        stamp = msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
        age = self.get_clock().now().nanoseconds/1e9-stamp
        try:
            if stamp < 0 or age > self.args.max_age or age < -0.05:
                raise ValueError("LF capture timestamp is stale or clocks are not synchronized")
            if msg.header.frame_id != self.c["frame_id"]:
                raise ValueError("Input frame_id must match calibrated center optical frame")
            x = unpack_grid(msg, self.c["height"], self.c["width"])
            pred = self.runner(x)[0]
            if not np.isfinite(pred).all():
                raise ValueError("Nonfinite model output")
            depth, valid, _ = navigation_outputs(pred[0], pred[1], self.c, max_scale=self.args.max_scale,
                                                 max_depth=self.args.max_depth)
            xyz = point_cloud(depth, self.c, self.args.point_stride)
            cloud = create_cloud_xyz32(msg.header, xyz)
            if self.get_clock().now().nanoseconds/1e9-stamp > self.args.max_age:
                raise ValueError("Geometry became stale during processing; dropping frame")
            self.disp_pub.publish(self.image(pred[0].astype('<f4'), '32FC1', msg.header))
            self.scale_pub.publish(self.image(np.exp(pred[1]).astype('<f4'), '32FC1', msg.header))
            self.depth_pub.publish(self.image(depth.astype("<f4"), "32FC1", msg.header))
            self.valid_pub.publish(self.image(valid.astype(np.uint8)*255, "mono8", msg.header))
            self.cloud_pub.publish(cloud)
            c = self.c
            info = CameraInfo()
            info.header, info.width, info.height = msg.header, c["width"], c["height"]
            info.distortion_model, info.d = "plumb_bob", [0.]*5
            info.k = [float(c["fx"]), 0., float(c["cx"]), 0., float(c["fy"]), float(c["cy"]), 0., 0., 1.]
            info.r = [1., 0., 0., 0., 1., 0., 0., 0., 1.]
            info.p = [float(c["fx"]), 0., float(c["cx"]), 0., 0., float(c["fy"]), float(c["cy"]), 0., 0., 0., 1., 0.]
            self.info_pub.publish(info)
            coverage = float(valid.mean())
            values = dict(valid_pct=coverage*100, points=len(xyz), processing_ms=(time.perf_counter()-start)*1000,
                          capture_age_ms=(self.get_clock().now().nanoseconds/1e9-stamp)*1000)
            good = coverage >= self.args.min_valid_fraction
            if good:
                self.last_success = self.get_clock().now().nanoseconds
            self.diagnostic(DiagnosticStatus.OK if good else DiagnosticStatus.WARN,
                            "Geometry available" if good else "Low valid coverage; unknown space", values)
            if time.monotonic()-self.last_log > 2:
                self.get_logger().info(json.dumps(values))
                self.last_log = time.monotonic()
        except (ValueError, RuntimeError) as e:
            self.diagnostic(DiagnosticStatus.ERROR, str(e))
            self.get_logger().error(str(e))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LFGeometryNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
