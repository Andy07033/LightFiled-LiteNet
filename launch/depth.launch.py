from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

def generate_launch_description():
    defaults = {'calibration': '', 'provider': 'auto', 'use_sim_time': 'false', 'max_age': '1.0'}
    return LaunchDescription([
        *[DeclareLaunchArgument(k, default_value=v) for k, v in defaults.items()],
        Node(package='lightfield_litenet', executable='depth_node', output='screen',
             parameters=[{'calibration': LaunchConfiguration('calibration'),
                          'provider': LaunchConfiguration('provider'),
                          'use_sim_time': ParameterValue(LaunchConfiguration('use_sim_time'), value_type=bool),
                          'max_age': ParameterValue(LaunchConfiguration('max_age'), value_type=float)}])])
