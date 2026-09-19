import launch
import launch_ros
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # 获取默认的urdf路径
    urdf_package_path = get_package_share_directory("robot_description")
    # urdf_path = os.path.join(urdf_package_path, "urdf", "my_robot.urdf")
    xacro_path = os.path.join(
        urdf_package_path, "urdf", "light_robot", "lightfieldrobot.urdf.xacro"
    )
    default_urdf_path = os.path.join(
        urdf_package_path, "config", "display_robot_model.rviz"
    )
    # 声明一个urdf名字参数，方便修改,寻找路径
    action_declare_arg_mode_path = launch.actions.DeclareLaunchArgument(
        name="model",
        default_value=xacro_path,
        description="Absolute path to robot urdf file",
    )

    # 通过文件路径，获取内容，并转换成参数值对象，以供传入robot_state_publisher节点
    # substitutions_command_result = launch.substitutions.Command(
    #     ["cat ", launch.substitutions.LaunchConfiguration("model")]
    # )

    substitutions_command_result = launch.substitutions.Command(
        ["xacro ", launch.substitutions.LaunchConfiguration("model")]
    )
    robot_description_value = launch_ros.parameter_descriptions.ParameterValue(
        substitutions_command_result, value_type=str
    )
    # 节点启动
    action_robot_state_publisher = launch_ros.actions.Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description_value}],
    )

    # 启动可动关节发布节点
    action_joint_state_publisher = launch_ros.actions.Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
    )

    rviz_node = launch_ros.actions.Node(
        package="rviz2",
        executable="rviz2",
        arguments=["-d", default_urdf_path],
    )

    return launch.LaunchDescription(
        [
            action_declare_arg_mode_path,
            action_robot_state_publisher,
            action_joint_state_publisher,
            rviz_node,
        ]
    )
