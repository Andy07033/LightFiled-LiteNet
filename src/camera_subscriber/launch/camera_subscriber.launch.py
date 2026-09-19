import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_params_file = os.path.join(
        get_package_share_directory("camera_subscriber"),
        "config",
        "camera_subscriber.yaml",
    )

    declared_arguments = [
        DeclareLaunchArgument(
            "params_file",
            default_value=default_params_file,
            description="参数文件路径",
        ),
        DeclareLaunchArgument("camera_count", default_value="9", description="相机路数"),
        DeclareLaunchArgument("topic_prefix", default_value="camera_", description="话题前缀"),
        DeclareLaunchArgument(
            "topic_suffix", default_value="/image_raw", description="话题后缀"
        ),
        DeclareLaunchArgument(
            "use_sim_time", default_value="true", description="是否使用 Gazebo 仿真时间"
        ),
    ]

    camera_subscriber_node = Node(
        package="camera_subscriber",
        executable="camera_subscriber_node",
        name="camera_subscriber",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {
                "camera_count": ParameterValue(
                    LaunchConfiguration("camera_count"), value_type=int
                ),
                "topic_prefix": LaunchConfiguration("topic_prefix"),
                "topic_suffix": LaunchConfiguration("topic_suffix"),
                "use_sim_time": ParameterValue(
                    LaunchConfiguration("use_sim_time"), value_type=bool
                ),
            },
        ],
    )

    return LaunchDescription(declared_arguments + [camera_subscriber_node])
