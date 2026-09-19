# 光场链路：9 路图像 → LiteNet → 深度/点云 → rviz2

推理代码已经从 `lightfield_bridge/lightfield_litenet/lfnav` 提取进本包，
模型与权重也搬进了 `models/`。**本包现在是自包含的**：不再需要 `lightfield_litenet`
那个 ROS 包，`lightfield_bridge` 可以作为纯训练仓库退役。

## 链路

```
Gazebo 9x libgazebo_ros_camera
        │  /camera_0..8/image_raw   512x512 rgb8, 10 Hz, best_effort
        ▼
  lightfield_node（本包，Python）
        │  1) 取 9 路 RGB -> 拼成 1536x1536 行优先大图
        │  2) 展开成 [1,27,512,512] float32 [0,1]（视角优先 RGB）
        │  3) onnxruntime 推理 -> [1,2,512,512] = 视差 + log 误差尺度
        │  4) Z = fx*B/(sign*d + offset)，反投影成 XYZ
        ├── /lf/disparity        32FC1
        ├── /lf/uncertainty_scale 32FC1（Laplace 视差尺度，不是置信度）
        ├── /lf/depth            32FC1 米制 Z，无效处 NaN
        ├── /lf/valid            mono8 255/0
        ├── /lf/camera_info      中心视角内参
        ├── /lf/points           PointCloud2（中心光学系，x右 y下 z前）
        ├── /lf/image_grid       rgb8 3x3 拼图（publish_grid 打开时，用来检查视角顺序）
        └── /diagnostics         延迟与有效覆盖率
        ▼
      rviz2（PointCloud2 /lf/points，Best Effort，Fixed Frame base_link）
```

## 为什么推理用 Python/onnxruntime，而不是 C++/cv::dnn

`src/litenet_node.cpp` 是你之前写的 C++/cv::dnn 草稿，我没有沿用它，理由是算量：

本网络输入 `[1,27,512,512]`，1,920,002 参数。按每层 MAC 估一遍，光 512×512 分辨率上的
四个方向 stem（9→12 两个 3×3 层）就是约 2.4 GMAC，加编解码与头部合计**单帧约 5 GMAC（≈10 GFLOP）**。
10 Hz 意味着 100 GFLOP/s —— Ubuntu 的 `libopencv-dev 4.5.4` 是无 CUDA 构建，
`cv::dnn` 只能 CPU，实测大概率只有零点几 Hz；而且 CPU 上它比 onnxruntime 还慢一截。

要走到实时，只有 `onnxruntime-gpu` 这条路，而它只在 Python 侧可用。
另外 `lfnav` 的数值代码是原作者已经做过数值与接口测试的，直接搬过来比重新实现可靠。

所以：**Python 负责推理，C++ 那部分保持你已经写好的订阅库不动。**

## 文件布局与来源对照

| 新位置 | 来源 / 说明 |
|---|---|
| `camera_subscriber/calibration.py` | ← `lfnav/ros_calibration.py`，逻辑等价 |
| `camera_subscriber/geometry.py` | ← `lfnav/geometry.py`，只留 `disparity_to_depth` / `navigation_outputs` / `point_cloud`；PFM、HCI 标定、训练评估留在原仓库 |
| `camera_subscriber/inference.py` | ← `lfnav/runtime.py`，`OnnxRunner` 逐字等价；新增 9 路 Image→RGB、拼图、`[1,27,H,W]` 展开 |
| `camera_subscriber/lightfield_node.py` | ← `lfnav/ros_node.py`，两处改动见下 |
| `scripts/lightfield_node` | 新的 `ros2 run` 入口（`#!/usr/bin/python3`，见下方说明） |
| `models/best.onnx` | 模型权重，由 `apply.sh` 从 `lightfield_bridge/.../weights/` 拷入 |
| `models/best.pt` | 训练权重，一并搬入但**不**安装进 `share/` |
| `config/lf_camera_sim.json` | 新增，仿真标定（理论值） |
| `config/lightfield_points.rviz` | 新增，带 PointCloud2 `/lf/points` |
| `launch/lightfield_pipeline.launch.py` | 新增，仿真＋推理＋rviz2 |

`lightfield_node.py` 相对 `ros_node.py` 的两处改动：

1. **直接订阅 9 路 `/camera_N/image_raw` 并在节点内拼图**，不再依赖外部先发 `/lf/image_grid`。
2. **推理跑在独立 callback group 的定时器里**。原版在订阅回调里直接推理；本网络 CPU 上
   单帧几百毫秒到数秒，那样会把回调线程堵死。现在永远只处理「最新一组同步帧」，
   慢就丢帧，不会让队列堆积。输出接口（话题名/类型/编码）与原版完全一致。

## 标定：`config/lf_camera_sim.json`

| 字段 | 值 | 来源 |
|---|---|---|
| `width`,`height` | 512 | `gazebo_sensor_plugin.xacro` 的 `<image>`；模型固定要求 512 |
| `fx`,`fy` | 274.53 | `HFOV=1.5009831567 rad` → `fx = 256 / tan(HFOV/2)`，方形像元 |
| `cx`,`cy` | 256.0 | Gazebo 取 `width/2` |
| `selected_baseline_m` | 0.006 | 相邻相机实际间距 = `2 × dense_step = 2 × 3 mm` |
| `angular_step` | 2 | 本 ONNX 固定要求 2，`baseline_m = 0.006/2 = 0.003` |
| `disparity_offset_px` | 0.0 | 仿真相机无零视差平面偏置 |
| `disparity_sign` | 1 | **唯一推不出来的量**，见下 |
| `frame_id` | `camera_4_optical_link` | 必须与拼图 header 的 frame_id、URDF frame 一致 |

`disparity_sign` 判据：**填反时 `denom` 全负，`/lf/depth` 整幅 NaN，`/diagnostics` 的
`valid_pct≈0`**。改成 `-1` 重启即可。`fx/cx` 可用 `/lf/camera_info` 核对（节点是用同
一套参数发的，所以要核对就对着 Gazebo 自己的 `/camera_4/camera_info` 看）。

## 构建

```bash
cd ~/ros2_ws
conda deactivate                      # 必须：否则 CMake 会抓到 miniconda 的 python3
source /opt/ros/humble/setup.bash

colcon build --packages-select camera_subscriber --symlink-install \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
source install/setup.bash
```

`--symlink-install` 让你改 `camera_subscriber/*.py` 不用重新 build。
注意 `models/best.onnx` 必须在 build 前就位，否则 install 阶段会报找不到文件。

### 推理依赖

```bash
sudo apt install -y python3-pip
# 一定用系统解释器：/usr/bin/python3 是 lightfield_node 的 shebang 指定的那个
/usr/bin/python3 -m pip install --user onnxruntime   # GPU 机器改装 onnxruntime-gpu
/usr/bin/python3 -m pip install --user "numpy<2"     # 重要，见下
/usr/bin/python3 -c "import onnxruntime, numpy, cv2; print(onnxruntime.__version__, numpy.__version__, cv2.__version__)"
```

**`onnxruntime` 会把 `numpy` 拉到 2.x，必须再按回去**：本机实测装完
`onnxruntime 1.23.2` 后 `numpy` 变成 2.2.6，系统自带的 `cv2 4.5.4`（还有依赖它的
`cv_bridge`）立刻报 `_ARRAY_API not found` / `numpy.core.multiarray failed to import`。
装成 `numpy 1.26.4` 后 `onnxruntime`、`cv2`、`cv_bridge` 三者都正常。别装 numpy 2。

入口脚本的 shebang 写死了 `/usr/bin/python3`：本机 `.bashrc` 会自动激活 conda base，
用 `/usr/bin/env python3` 有可能落到 miniconda 解释器上（那里没有 onnxruntime、
甚至可能没有 rclpy）。所以别改成 `env python3`。

## 运行

```bash
source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash
ros2 launch camera_subscriber lightfield_pipeline.launch.py
```

拆开调试（推荐第一次这样跑，日志分开看得清）：

```bash
# 终端 1：仿真
ros2 launch robot_description gazebo_sim.launch.py
# 终端 2：推理（用 launch 传默认标定与权重绝对路径）
ros2 launch camera_subscriber lightfield_pipeline.launch.py rviz:=false
# 终端 3：rviz2
rviz2 -d ~/ros2_ws/install/camera_subscriber/share/camera_subscriber/config/lightfield_points.rviz
```

也可以直接 `ros2 run`，但必须显式给两个绝对路径：

```bash
ros2 run camera_subscriber lightfield_node --ros-args \
  -p calibration:=$HOME/ros2_ws/install/camera_subscriber/share/camera_subscriber/config/lf_camera_sim.json \
  -p model_path:=$HOME/ros2_ws/install/camera_subscriber/share/camera_subscriber/models/best.onnx \
  -p provider:=cpu -p use_sim_time:=true
```

## 逐级验证

```bash
ros2 topic hz /camera_0/image_raw                 # ~10 Hz，采集正常
ros2 topic hz /lf/depth                           # 推理速率：CPU 上可能就是零点几 Hz
ros2 topic echo /diagnostics                      # valid_pct / processing_ms / capture_age_ms
ros2 topic echo /lf/points --once --field header  # frame_id 应是 camera_4_optical_link
ros2 topic info /lf/points -v                     # 发布/订阅都应是 BEST_EFFORT
rqt_image_view /lf/image_grid                     # 看九宫格视角顺序对不对（应为行优先）
rqt_image_view /lf/valid                          # 有效区域
```

rviz2 配置里已含 Grid、**PointCloud2 `/lf/points`**、RobotModel、TF，
Fixed Frame = `base_link`（`camera_4_optical_link` 的 TF 由 `robot_state_publisher` 提供）。

**先看 `/diagnostics` 的 `processing_ms` 再谈实时**：如果 CPU 上只有 0.3 Hz，
`capture_age_ms` 会贴着 `max_age`（默认 1 s），此时点云仍能显示，但别拿它导航。
要提速就换 GPU：`provider:=cuda` + `onnxruntime-gpu`；或把 `point_stride` 调大只减压点云，不影响推理。

## 排障顺序

1. **节点起不来** → 依次看是不是
   `ModuleNotFoundError: onnxruntime`（装到系统 python3 了吗）、
   `必须设置 calibration / model_path`（用 `ros2 run` 时没给绝对路径）、
   `权重输入形状 ... 与标定 ... 不匹配`。
2. **一直报「采集时间戳过期或时钟不同步」** → `use_sim_time` 没开，或 Gazebo 没发 `/clock`。
3. **报 frame_id 不匹配** → 拼图的 frame_id 与标定 `frame_id` 不一致；
   仿真里应为 `camera_4_optical_link`。
4. **`/lf/depth` 全 NaN、`valid_pct≈0`** → `disparity_sign` 反了，改成 `-1`。
5. **深度尺度不对** → 把机器人放到离墙**已知距离**处比对 `/lf/depth` 中心区域中位数；
   偏差是固定倍数时优先怀疑 `selected_baseline_m`（0.006）与 `fx`（274.53）。
   或者直接跑离线自检（不用开 Gazebo）：
   ```bash
   /usr/bin/python3 ~/ros2_ws/src/camera_subscriber/scripts/offline_check.py -z 0.5 1.0 2.0
   ```
   看 `Z_est` 列是否等于 `Z_true` 列。本机实测：0.5/1.0/2.0 m 的 `Z_est` 为
   0.487/0.986/1.991 m，说明符号与基线换算都对；`d_pred_px` 那一列在远距离会明显偏小，
   那是网络在亚像素视差下的固有偏差，不是标定错。
6. **`ros2 run gazebo_ros spawn_entity.py` 报 `No module named 'lxml'`（或 `_rclpy_pybind11` 找不到）**
   → 这个脚本的 shebang 是 `#!/usr/bin/env python3`，如果你的终端自动激活了 conda
   （或工作区里的 venv，本机 `~/ros2_ws/venv` 是 conda 派生的 python 3.14），
   `python3` 会落到那个解释器上。先 `conda deactivate` / `deactivate`，或者显式：
   ```bash
   /usr/bin/python3 /opt/ros/humble/lib/gazebo_ros/spawn_entity.py -topic robot_description -entity my_robot
   ```
   本包的 `lightfield_node` 不受影响：它的 shebang 写死了 `/usr/bin/python3`。
7. **话题列表空、节点都在跑却收不到** → WSL2 镜像网络下 FastDDS 老问题：
   ```bash
   rm -f /dev/shm/fastrtps_* /dev/shm/sem.fastrtps_*
   export ROS_LOCALHOST_ONLY=1
   # 仍不行用纯 UDP 配置：
   export FASTRTPS_DEFAULT_PROFILES_FILE=$(ros2 pkg prefix camera_subscriber)/share/camera_subscriber/config/fastdds_udp_only.xml
   ```
   （别用 `kill -9` 杀 ROS 节点，会在 `/dev/shm` 留残留。）

## 与 `lightfield_bridge` 的关系

`~/ros2_ws/lightfield_bridge/lightfield_litenet` 现在只剩训练价值（`train.py`、`lfnav/model.py`、
`configs/`、`assets/`）。它不再被本链路依赖，但 `colcon build` 仍会连它一起构建。
要退役它又不删代码，加一个 `COLCON_IGNORE` 即可：

```bash
touch ~/ros2_ws/lightfield_bridge/COLCON_IGNORE
rm -rf ~/ros2_ws/build/lightfield_litenet ~/ros2_ws/install/lightfield_litenet
```

（想继续用它的 `depth_node` + 旧 `/lf/image_grid` 路线也没问题，两条线互不冲突，
只是别再让两个节点同时发 `/lf/depth`、`/lf/points`。）

## 已知问题 / 待办

- 标定是**理论值**，未做实测标定；`disparity_sign` 与深度尺度已用离线自检核对为自洽
  （见下），真实标定仍应实测。
- `point_stride` 默认 2（256×256 ≈ 65k 点/帧）；改 1 是 262k 点/帧，注意 rviz 与总线压力。
- 本包 README 里两条老坑依旧适用：**订阅端必须 best_effort**、
  **构建设置不要让 conda 抢走 python3**。

## 本机实测：端到端跑通记录（2026-09-16）

环境：WSL2 Ubuntu 22.04 + ROS 2 Humble + Gazebo classic 11，**无界面 gzserver、无独立 GPU**，
机器人放到距墙 0.762 m 处正对墙面。

| 项 | 实测 |
|---|---|
| 9 路相机 | `/camera_0..8/image_raw`，512×512 rgb8，约 9 Hz，`frame_id=camera_4_optical_link` |
| 推理 | CPU 单帧 **1.9~2.3 s（≈0.5 Hz）**；`/diagnostics` 里 `valid_pct`、`processing_ms` 正常上报 |
| 发布 | `/lf/depth`、`/lf/valid`、`/lf/points`（`frame_id=camera_4_optical_link`）都有数据 |
| 点云 | `point_stride=2` 时 5k~14k 点，z 中位与 `/lf/depth` 一致 |

关键发现，按重要性排：

1. **场景必须有纹理，否则深度是错的。** 原 `my_world.world` 的墙/地面都是 `Gazebo/Grey`
   （整幅图 |grad|≈1.7，几乎没有特征），实测 `valid_pct≈8%`、深度中位 **2.08 m**，
   而真实距离 0.76 m —— 网络在没有纹理时输出的是瞎猜值。把材质换成
   `Gazebo/Bricks` 后（可在 `world/my_world.world` 里把 `<name>Gazebo/Grey</name>`
   改成 `<name>Gazebo/Bricks</name>`），`valid_pct` 升到 **21.5%**、深度中位
   **0.872 m**（相对 0.762 m 偏差约 14%），点云位置正确。真实感/纹理越丰富，越接近
   训练分布（HCI 自然图像），深度越可信。
2. **有效距离集中在近距（约 0.3~1.5 m）。** 6 mm 孔径下，1 m 处相邻视角视差只有
   0.82 px、2 m 处 0.41 px，网络自带的 Laplace 尺度在这个量级会超过
   `max_scale`/`max_relative_depth_error` 阈值，于是被判无效（1 m 时约 26% 有效、
   2 m 时 15%）。远处想用就得放宽这两个阈值（代价是噪声点变多），或者加大相机间距
   并同步改 `selected_baseline_m`。
3. **9 路同步不能想当然。** 正常实时率下 Gazebo 会给同一仿真步的所有相机盖同一个时间戳；
   但本机（9 路同时软件渲染 + CPU 推理抢满 4 线程）实测时间戳跨度中位 **0.2 s**、最大 0.9 s，
   旧默认 `sync_tolerance_sec=0.02` 会让节点一组都收不下、而且只表现为"不出结果"。
   现在默认放宽到 0.05 s，并且超过阈值会每 5 s 打一条 WARN（带实测跨度与建议），
   `camera_subscriber.yaml` 里同一参数也已同步。
4. **CPU 只有 0.5 Hz，`max_age`（默认 1 s）会开始丢帧。** 想实时只有
   `onnxruntime-gpu` + `provider:=cuda` 一条路；换 GPU 后 `point_stride` 也可以调回 1。
5. rviz2 用 `config/lightfield_points.rviz`（Fixed Frame `base_link`、PointCloud2
   `/lf/points`、Reliability `Best Effort`）即可看到点云；`camera_4_optical_link` 的 TF
   由 `robot_state_publisher` 提供，不需要额外发静态 TF。
