"""整条光场链路：Gazebo 仿真 -> 9 路图像 -> LiteNet 深度/点云 -> rviz2。

    ros2 launch camera_subscriber lightfield_pipeline.launch.py

只要仿真和 rviz2（推理手动另开终端）时，用 rviz:=false。
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("camera_subscriber")
    robot_share = get_package_share_directory("robot_description")

    default_params_file = os.path.join(share, "config", "camera_subscriber.yaml")
    default_calibration = os.path.join(share, "config", "lf_camera_sim.json")
    default_model = os.path.join(share, "models", "best.onnx")
    default_rviz_config = os.path.join(share, "config", "lightfield_points.rviz")

    declared_arguments = [
        DeclareLaunchArgument(
            "params_file", default_value=default_params_file,
            description="相机路数/话题名/QoS 等参数文件"),
        DeclareLaunchArgument(
            "calibration", default_value=default_calibration,
            description="标定 JSON（绝对路径）"),
        DeclareLaunchArgument(
            "model_path", default_value=default_model,
            description="ONNX 权重路径"),
        DeclareLaunchArgument(
            "provider", default_value="auto",
            description="onnxruntime 执行后端：auto / cpu / cuda"),
        DeclareLaunchArgument(
            "max_age", default_value="1.0",
            description="允许的最大采集延迟（秒），超过就判为过期"),
        DeclareLaunchArgument(
            "point_stride", default_value="2",
            description="点云采样步长：2 -> 256x256 点，1 -> 512x512 点"),
        DeclareLaunchArgument(
            "publish_grid", default_value="true",
            description="是否把拼好的 3x3 大图发到 /lf/image_grid 便于检查"),
        DeclareLaunchArgument(
            "use_sim_time", default_value="true", description="是否使用 Gazebo 仿真时间"),
        DeclareLaunchArgument("rviz", default_value="true", description="是否启动 rviz2"),
        DeclareLaunchArgument(
            "rviz_config", default_value=default_rviz_config, description="rviz2 配置文件"),
    ]

    use_sim_time = ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)

    # 1) 仿真：Gazebo + robot_state_publisher + 9 路相机插件
    gazebo_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(robot_share, "launch", "gazebo_sim.launch.py"))
    )

    # 2) 推理：订阅 9 路 -> 拼图 -> ONNX -> /lf/depth、/lf/points
    lightfield_node = Node(
        package="camera_subscriber",
        executable="lightfield_node",
        name="lightfield_node",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "calibration": LaunchConfiguration("calibration"),
                "model_path": LaunchConfiguration("model_path"),
                "provider": LaunchConfiguration("provider"),
                "max_age": ParameterValue(LaunchConfiguration("max_age"), value_type=float),
                "point_stride": ParameterValue(
                    LaunchConfiguration("point_stride"), value_type=int),
                "publish_grid": ParameterValue(
                    LaunchConfiguration("publish_grid"), value_type=bool),
                "use_sim_time": use_sim_time,
            },
        ],
    )

    # 3) rviz2：默认打开带 PointCloud2 的配置
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", LaunchConfiguration("rviz_config")],
        parameters=[{"use_sim_time": use_sim_time}],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(
        declared_arguments + [gazebo_sim, lightfield_node, rviz_node]
    )
