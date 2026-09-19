# camera_subscriber

订阅 3×3 光场相机阵列的 9 路图像话题，并为后续接入模型预留接口。

> 光场推理（LiteNet ONNX → 深度/点云 → rviz2）已经并进本包，见
> [LIGHTFIELD_PIPELINE.md](LIGHTFIELD_PIPELINE.md)：移植来源对照表、标定含义、
> 构建/运行/排障，以及本机端到端实测结果都在那里。
> 一句话跑法：
>
> ```bash
> ros2 launch camera_subscriber lightfield_pipeline.launch.py        # 仿真 + 推理 + rviz2
> ros2 launch camera_subscriber lightfield_pipeline.launch.py rviz:=false   # 推理单独调试
> ```
>
> 本文档剩下的部分讲的是纯 C++ 订阅节点（`camera_subscriber_node`），
> 光场链路不依赖它。

## 话题

话题名由 `robot_description/urdf/light_robot/plugins/gazebo_sensor_plugin.xacro` 里
`libgazebo_ros_camera.so` 的 `camera_name=camera_${index}` 决定。

| index | 话题 | xacro 中的 (u, v) | 行列 |
|---|---|---|---|
| 0 | `/camera_0/image_raw` | (-2, -2) | row 0, col 0 |
| 1 | `/camera_1/image_raw` | (-2,  0) | row 0, col 1 |
| 2 | `/camera_2/image_raw` | (-2,  2) | row 0, col 2 |
| 3 | `/camera_3/image_raw` | ( 0, -2) | row 1, col 0 |
| 4 | `/camera_4/image_raw` | ( 0,  0) 中心相机 | row 1, col 1 |
| 5 | `/camera_5/image_raw` | ( 0,  2) | row 1, col 2 |
| 6 | `/camera_6/image_raw` | ( 2, -2) | row 2, col 0 |
| 7 | `/camera_7/image_raw` | ( 2,  0) | row 2, col 1 |
| 8 | `/camera_8/image_raw` | ( 2,  2) | row 2, col 2 |

- 消息类型：`sensor_msgs/msg/Image`，512×512，10 Hz
- 每路另有 `/camera_N/camera_info`（`sensor_msgs/CameraInfo`），本包暂不订阅
- 索引是**行优先**排列（`index = row*3 + col`），与 xacro 里 (u, v) 的顺序严格对应。
  做光场重建/视差时**不要重排索引**，否则子孔径的几何关系会错。

## 编译

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select camera_subscriber
source install/setup.bash
```

### ⚠️ 坑：conda 会抢走 CMake 的 Python

如果终端里 conda base 是自动激活的（提示符带 `(base)`），`ament_cmake_core` 的
`find_package(Python3 REQUIRED COMPONENTS Interpreter)` 会找到 `~/miniconda3/bin/python3`，
而 conda 里没有 `catkin_pkg`，于是报：

```
ModuleNotFoundError: No module named 'catkin_pkg'
CMake Error at .../ament_package_xml.cmake:95 (message):
  execute_process(/home/yaowei/miniconda3/bin/python3 .../package_xml_2_cmake.py ...)
```

三种修法，任选其一：

```bash
# 1) 构建时显式钉死解释器（推荐，不动环境）
colcon build --packages-select camera_subscriber \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3

# 2) 持久化到 colcon 默认参数，之后不用每次加
mkdir -p ~/.colcon
cat > ~/.colcon/defaults.yaml <<'YAML'
{
  "build": {
    "cmake-args": ["-DPython3_EXECUTABLE=/usr/bin/python3"]
  }
}
YAML

# 3) 干脆别让 conda 自动激活 base
conda config --set auto_activate_base false   # 重开终端生效
```

注意：已缓存的 `build/camera_subscriber/CMakeCache.txt` 里记着错的解释器路径，
换环境后要 `rm -rf build/camera_subscriber install/camera_subscriber` 再构建。

`colcon` 本身不受影响（`/usr/bin/colcon` 的 shebang 就是 `#!/usr/bin/python3`）。

## 运行

```bash
# 终端 1：Gazebo 仿真（9 路相机插件在里面）
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 launch robot_description gazebo_sim.launch.py

# 终端 2：订阅节点
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 launch camera_subscriber camera_subscriber.launch.py
```

也可以用不方便带 launch 的方式快速试：

```bash
ros2 run camera_subscriber camera_subscriber_node \
  --ros-args -p camera_count:=9 -p stats_period_sec:=2.0
```

节点每 `stats_period_sec` 秒打印一次 9 路状态，某路超时或没数据会额外 `WARN`：

```
[INFO] [camera_subscriber]: 相机订阅状态（9 路）：
  [0] camera_0/image_raw -> 10.0 Hz，累计 300 帧，512x512 rgb8，距上一帧 0.0 s
  [1] camera_1/image_raw -> 10.0 Hz，累计 300 帧，512x512 rgb8，距上一帧 0.0 s
  ...
```

## 参数

| 参数 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `camera_count` | int | 9 | 相机路数 |
| `topic_prefix` | string | `camera_` | 话题前缀 |
| `topic_suffix` | string | `/image_raw` | 话题后缀 |
| `qos_reliability` | string | `best_effort` | `best_effort` / `reliable` |
| `qos_depth` | int | 5 | 队列深度 |
| `stats_period_sec` | double | 5.0 | 统计打印周期 |
| `stale_timeout_sec` | double | 1.0 | 超时告警阈值 |
| `sync_tolerance_sec` | double | 0.02 | 判定"同一时刻"的时间戳跨度上限 |

话题名 = `topic_prefix + index + topic_suffix`。

**QoS 是最大的坑**：`gazebo_ros_camera` 用 `SensorDataQoS`（`BEST_EFFORT`）发布。
订阅端若用 `RELIABLE`，QoS 不兼容，**一帧都收不到，而且不报任何错**。
所以默认 `best_effort`。用 `ros2 topic info <话题> --verbose` 可以核对两端的 Reliability。

## 接入模型

### 方式一：整组回调（推荐）

9 路都到齐、且时间戳跨度 ≤ `sync_tolerance_sec` 时回调一次，直接喂模型。
改 `src/camera_subscriber_main.cpp` 里的 `setFrameSetCallback`：

```cpp
node->setFrameSetCallback(
  [](const std::vector<camera_subscriber::ImagePtr> & frames) {
    // frames[0..8] 依次是 camera_0..camera_8，ConstSharedPtr，不拷贝像素
    auto input  = buildTensor(frames);   // 9 路 -> NCHW / NHWC
    auto output = model_->infer(input);
    publishResult(output);
  });
```

### 方式二：主动拉取最新帧

模型跑在自己的线程/主循环里时用这个：

```cpp
std::vector<camera_subscriber::ImagePtr> frames;
if (node->latestFrames(frames)) {          // 9 路全有才返回 true
  // frames.size() == camera_count
}
auto one = node->latestFrame(4);           // 只要中心相机
```

### 转 cv::Mat

`CMakeLists.txt` 加 `find_package(cv_bridge REQUIRED)` 和
`ament_target_dependencies(... cv_bridge)`，`package.xml` 加 `<depend>cv_bridge</depend>`：

```cpp
#include <cv_bridge/cv_bridge.h>
cv::Mat bgr = cv_bridge::toCvShare(frames[i], "bgr8")->image;  // 零拷贝
```

`toCvShare` 的 Mat 指向 `frames[i]` 的像素，帧释放后即悬空；要长期保存必须 `.clone()`。

### 性能提示

9 路 × 512×512×3 ≈ 7 MB/帧，10 Hz ≈ 70 MB/s。别在订阅回调里做重活——
`setFrameSetCallback` 已经在锁外执行；数据量再大就用 `latestFrames()` 主动拉，
靠 `qos_depth` 丢帧，而不是让队列堆积。

## 校验命令

```bash
ros2 topic list | grep image_raw                  # 9 路话题
ros2 topic info /camera_0/image_raw --verbose      # 核对两端 QoS
ros2 topic hz /camera_0/image_raw                  # 实测帧率（应为 10 Hz）
ros2 node info /camera_subscriber                  # 确认订阅了 9 路
ros2 param list /camera_subscriber
```

## 结构

```
camera_subscriber/
├── CMakeLists.txt
├── package.xml
├── include/camera_subscriber/camera_subscriber_node.hpp
├── src/
│   ├── camera_subscriber_node.cpp     # 订阅逻辑（编译成 camera_subscriber_core 库）
│   └── camera_subscriber_main.cpp     # main + 模型接入点
├── launch/camera_subscriber.launch.py
└── config/camera_subscriber.yaml
```

订阅逻辑做成库（`camera_subscriber_core`）并导出了头文件，写模型节点时可以
`find_package(camera_subscriber REQUIRED)` 后直接复用，不必复制订阅代码。

## ⚠️ WSL2 下 DDS 发现失效（"进程都在跑但收不到数据"）

本机是 WSL2 **镜像网络**模式（`eth0`~`eth3` 全 DOWN，只有 `eth4` 承载流量），
FastDDS 默认依赖 UDP 组播 + 共享内存做发现，在这种环境下会出现
**节点都活着、话题却始终匹配不上**：`ros2 topic list` 是空的，
订阅端一直打"还没收到任何帧"，而两边都不报任何错。
本机还开着 `avahi-daemon`，同样会干扰组播。

按顺序排查：

```bash
# 1) 清掉被 kill -9 留下的 FastDDS 共享内存残留（最常见的元凶）
rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_*

# 2) 只在本机通信时，强制走 loopback
export ROS_LOCALHOST_ONLY=1

# 3) 仍不行：用纯 UDPv4 配置绕开共享内存与组播（本包已带此文件）
export FASTRTPS_DEFAULT_PROFILES_FILE=\
$(ros2 pkg prefix camera_subscriber)/share/camera_subscriber/config/fastdds_udp_only.xml

# 4) 都不行就重启 WSL 实例（Windows PowerShell 里执行，会关掉所有 WSL 进程）
#    wsl --shutdown
```

习惯上：**别用 `kill -9` 杀 ROS 节点**（用 Ctrl+C 或 `kill` 就够）。
`kill -9` 会在 `/dev/shm` 留下 `fastrtps_*` 残留，越积越多，最后发现彻底失效——
这个现象很容易被误判成"代码有 bug"。

代价：纯 UDP 关掉了共享内存零拷贝，9 路 512×512 图像会多几次内存拷贝，
CPU 占用会上升；本机发现正常时不必开。
