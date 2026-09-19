"""光场深度 / 点云节点。

由 lightfield_bridge/lightfield_litenet/lfnav/ros_node.py 改写，两处关键不同：

1. 直接订阅 9 路 ``/camera_N/image_raw``，在节点内部拼成 3x3 大图，
   不再依赖外部先发一张 ``/lf/image_grid``；
2. 推理跑在独立 callback group 的定时器里 —— 这个网络在 CPU 上单帧约 10 GFLOP，
   可能要几百毫秒到数秒，放在订阅回调里会把回调线程堵死。这样做的效果是
   「永远只处理最新一组同步帧」，慢就丢帧，而不是让队列堆积发霉。

输出话题与原版保持一致：
``/lf/disparity``、``/lf/uncertainty_scale``、``/lf/depth``、``/lf/valid``、
``/lf/camera_info``、``/lf/points``、``/diagnostics``；
另有 ``/lf/image_grid``（``publish_grid`` 为 true 时）方便检查视角顺序。
"""

import json
import time

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py.point_cloud2 import create_cloud_xyz32

from .calibration import load_calibration
from .geometry import navigation_outputs, point_cloud
from .inference import OnnxRunner, build_mosaic, image_msg_to_rgb, mosaic_to_input

NODE_NAME = "lightfield_node"


class LightfieldNode(Node):
    def __init__(self):
        super().__init__(NODE_NAME)

        # ------------------------- 参数 -------------------------
        self.camera_count = int(self.declare_parameter("camera_count", 9).value)
        topic_prefix = self.declare_parameter("topic_prefix", "camera_").value
        topic_suffix = self.declare_parameter("topic_suffix", "/image_raw").value
        reliability = self.declare_parameter("qos_reliability", "best_effort").value
        qos_depth = max(1, int(self.declare_parameter("qos_depth", 1).value))
        # 9 路必须是同一时刻的曝光。gazebo_ros_camera 正常情况下同一个仿真步里
        # 给所有传感器盖同一个时间戳；但仿真被拖慢（无 GPU、9 路同时渲染）时，
        # 各路会落到不同的仿真步上，跨度能到零点几秒。默认给半帧周期（10 Hz -> 0.05），
        # 再不行就由下面的告警提示你把这个参数调大。
        self.sync_tolerance = float(self.declare_parameter("sync_tolerance_sec", 0.05).value)
        self.max_age = float(self.declare_parameter("max_age", 1.0).value)
        work_period = float(self.declare_parameter("work_period_sec", 0.05).value)
        self.min_valid_fraction = float(self.declare_parameter("min_valid_fraction", 0.1).value)
        self.point_stride = int(self.declare_parameter("point_stride", 2).value)
        self.border = int(self.declare_parameter("border", 11).value)
        self.min_depth = float(self.declare_parameter("min_depth", 0.15).value)
        self.max_depth = float(self.declare_parameter("max_depth", 10.0).value)
        self.max_scale = float(self.declare_parameter("max_scale", 0.15).value)
        self.max_relative_depth_error = float(
            self.declare_parameter("max_relative_depth_error", 0.1).value)
        self.publish_grid = bool(self.declare_parameter("publish_grid", True).value)

        calibration_path = self.declare_parameter("calibration", "").value
        model_path = self.declare_parameter("model_path", "").value
        provider = self.declare_parameter("provider", "auto").value
        if not calibration_path:
            raise ValueError("必须设置 calibration（标定 JSON 的绝对路径）")
        if not model_path:
            raise ValueError("必须设置 model_path（best.onnx 的绝对路径）")

        self.c = load_calibration(calibration_path)
        self.runner = OnnxRunner(model_path, provider)
        self.get_logger().info("ONNX providers: %s" % (self.runner.providers,))
        if self.runner.providers[0] == "CPUExecutionProvider":
            self.get_logger().warning(
                "正在用 CPU 推理：本网络单帧约 10 GFLOP，处理速率可能只有零点几 Hz，"
                "先用 /diagnostics 的 processing_ms 看清再谈实时")
        expected = (1, 27, self.c["height"], self.c["width"])
        if tuple(self.runner.shape) != expected:
            raise ValueError("权重输入形状 %s 与标定 %s 不匹配" % (self.runner.shape, expected))

        # ------------------------- 发布 -------------------------
        # 点云/图像用 best-effort + 队列 1，和原版一致：宁可丢帧也别排队
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.disp_pub = self.create_publisher(Image, "/lf/disparity", qos)
        self.scale_pub = self.create_publisher(Image, "/lf/uncertainty_scale", qos)
        self.depth_pub = self.create_publisher(Image, "/lf/depth", qos)
        self.valid_pub = self.create_publisher(Image, "/lf/valid", qos)
        self.info_pub = self.create_publisher(CameraInfo, "/lf/camera_info", qos)
        self.cloud_pub = self.create_publisher(PointCloud2, "/lf/points", qos)
        self.grid_pub = self.create_publisher(Image, "/lf/image_grid", qos) if self.publish_grid else None
        self.diag_pub = self.create_publisher(DiagnosticArray, "/diagnostics", 10)

        # ------------------------- 订阅 -------------------------
        # gazebo_ros_camera 用 SensorDataQoS（BEST_EFFORT）发布，订阅端用 RELIABLE
        # 会 QoS 不兼容：一帧都收不到而且不报错
        sub_qos = QoSProfile(
            depth=qos_depth,
            reliability=(ReliabilityPolicy.RELIABLE if reliability == "reliable"
                         else ReliabilityPolicy.BEST_EFFORT),
            history=HistoryPolicy.KEEP_LAST)
        self.sub_group = MutuallyExclusiveCallbackGroup()
        self.frames = [None] * self.camera_count
        self.topics = ["%s%d%s" % (topic_prefix, i, topic_suffix) for i in range(self.camera_count)]
        for index, topic in enumerate(self.topics):
            self.create_subscription(
                Image, topic, lambda msg, i=index: self.on_image(i, msg),
                sub_qos, callback_group=self.sub_group)
        self.get_logger().info(
            "已订阅 %d 路：%s" % (self.camera_count, ", ".join(self.topics)))

        # ------------------------- 处理循环 -------------------------
        self.work_group = MutuallyExclusiveCallbackGroup()
        self.diag_group = MutuallyExclusiveCallbackGroup()
        self.last_stamp = -1.0
        self.last_success_ns = None
        self.last_error = ""
        self.last_log = 0.0
        self.last_sync_warn = 0.0
        self.sync_rejects = 0
        self.processed = 0
        self.create_timer(work_period, self.try_process, callback_group=self.work_group)
        self.create_timer(0.5, self.watchdog, callback_group=self.diag_group)

    # ------------------------------------------------------------------
    # 订阅侧：只记最新帧，不做任何重活
    # ------------------------------------------------------------------
    def on_image(self, index, msg):
        self.frames[index] = msg

    # ------------------------------------------------------------------
    # 处理侧
    # ------------------------------------------------------------------
    def try_process(self):
        frames = list(self.frames)
        if any(frame is None for frame in frames):
            return

        stamps = [f.header.stamp.sec + f.header.stamp.nanosec * 1e-9 for f in frames]
        spread = max(stamps) - min(stamps)
        if spread > self.sync_tolerance:
            self.sync_rejects += 1
            if time.monotonic() - self.last_sync_warn > 5.0:
                self.last_sync_warn = time.monotonic()
                self.get_logger().warning(
                    "9 路时间戳跨度 %.3f s 超过 sync_tolerance_sec=%.3f s，"
                    "已连续丢弃 %d 组（各路不是硬件同步时，把参数调到一帧周期左右，"
                    "例如 0.1 s；运动场景下这会带来运动误差）"
                    % (spread, self.sync_tolerance, self.sync_rejects))
            return

        center = frames[self.camera_count // 2]
        stamp = center.header.stamp.sec + center.header.stamp.nanosec * 1e-9
        now = self.get_clock().now().nanoseconds * 1e-9
        # 已经处理过的旧组直接跳过；但仿真时钟回退（重启仿真）时不能一直卡死，
        # 所以只在「时钟没有回退」的前提下才认定它是旧组
        if stamp <= self.last_stamp and now >= self.last_stamp:
            return
        if stamp < 0 or now - stamp > self.max_age or now - stamp < -0.05:
            self.fail("LF 采集时间戳过期或时钟不同步（检查 use_sim_time 与 /clock）")
            return
        if center.header.frame_id != self.c["frame_id"]:
            self.fail("输入 frame_id=%s 必须是标定里的中心光学坐标系 %s"
                      % (center.header.frame_id, self.c["frame_id"]))
            return

        self.last_stamp = stamp
        try:
            self.process(frames, center, stamp, now)
        except (ValueError, RuntimeError) as exc:
            self.fail(str(exc))

    def process(self, frames, center, stamp, now):
        started = time.perf_counter()
        height, width = self.c["height"], self.c["width"]

        rgb_frames = [image_msg_to_rgb(frame) for frame in frames]
        for image in rgb_frames:
            if image.shape != (height, width, 3):
                raise ValueError("图像形状 %s 与标定的 %dx%d 不符"
                                 % (image.shape, width, height))
        mosaic = build_mosaic(rgb_frames, height, width)
        prediction = self.runner(mosaic_to_input(mosaic, height, width))[0]
        if not np.isfinite(prediction).all():
            raise ValueError("模型输出存在非有限值")

        depth, valid, _ = navigation_outputs(
            prediction[0], prediction[1], self.c,
            min_depth=self.min_depth, max_depth=self.max_depth,
            max_scale=self.max_scale,
            max_relative_depth_error=self.max_relative_depth_error,
            border=self.border)
        xyz = point_cloud(depth, self.c, self.point_stride)
        cloud = create_cloud_xyz32(center.header, xyz)

        if self.get_clock().now().nanoseconds * 1e-9 - stamp > self.max_age:
            raise ValueError("推理期间数据已过期，丢弃这一帧")

        header = center.header
        self.disp_pub.publish(self.image(prediction[0].astype("<f4"), "32FC1", header))
        self.scale_pub.publish(self.image(np.exp(prediction[1]).astype("<f4"), "32FC1", header))
        self.depth_pub.publish(self.image(depth.astype("<f4"), "32FC1", header))
        self.valid_pub.publish(self.image(valid.astype(np.uint8) * 255, "mono8", header))
        if self.grid_pub is not None:
            self.grid_pub.publish(self.image(mosaic, "rgb8", header))
        self.cloud_pub.publish(cloud)
        self.info_pub.publish(self.camera_info(header))

        coverage = float(valid.mean())
        values = {
            "valid_pct": coverage * 100.0,
            "points": int(len(xyz)),
            "processing_ms": (time.perf_counter() - started) * 1000.0,
            "capture_age_ms": (now - stamp) * 1000.0,
        }
        good = coverage >= self.min_valid_fraction
        if good:
            self.last_success_ns = self.get_clock().now().nanoseconds
        self.diagnostic(
            DiagnosticStatus.OK if good else DiagnosticStatus.WARN,
            "几何可用" if good else "有效覆盖率过低；未知区域不能当作可通行", values)
        self.processed += 1
        if time.monotonic() - self.last_log > 2.0:
            self.get_logger().info("[#%d] %s" % (self.processed, json.dumps(values, ensure_ascii=False)))
            self.last_log = time.monotonic()

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def image(array, encoding, header):
        msg = Image()
        msg.header = header
        msg.height, msg.width = int(array.shape[0]), int(array.shape[1])
        msg.encoding, msg.is_bigendian = encoding, False
        channels = array.shape[2] if array.ndim == 3 else 1
        msg.step = msg.width * array.dtype.itemsize * channels
        msg.data = np.ascontiguousarray(array).tobytes()
        return msg

    def camera_info(self, header):
        c = self.c
        info = CameraInfo()
        info.header, info.width, info.height = header, int(c["width"]), int(c["height"])
        info.distortion_model, info.d = "plumb_bob", [0.0] * 5
        info.k = [float(c["fx"]), 0.0, float(c["cx"]),
                  0.0, float(c["fy"]), float(c["cy"]),
                  0.0, 0.0, 1.0]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [float(c["fx"]), 0.0, float(c["cx"]), 0.0,
                  0.0, float(c["fy"]), float(c["cy"]), 0.0,
                  0.0, 0.0, 1.0, 0.0]
        return info

    def diagnostic(self, level, message, values=None):
        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        status = DiagnosticStatus(level=level, name="lfnav/geometry", message=message,
                                  hardware_id=self.c["frame_id"])
        status.values = [KeyValue(key=str(k), value=str(v)) for k, v in (values or {}).items()]
        array.status = [status]
        self.diag_pub.publish(array)

    def fail(self, message):
        if message != self.last_error:
            self.get_logger().error(message)
            self.last_error = message
        self.diagnostic(DiagnosticStatus.ERROR, message)

    def watchdog(self):
        now_ns = self.get_clock().now().nanoseconds
        if (self.last_success_ns is None
                or (now_ns - self.last_success_ns) * 1e-9 > self.max_age
                or now_ns < self.last_success_ns):
            self.diagnostic(DiagnosticStatus.ERROR,
                            "没有新鲜的 LF 几何输出；导航层必须把未知区域当不可通行")


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LightfieldNode()
        executor = MultiThreadedExecutor(num_threads=3)
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
