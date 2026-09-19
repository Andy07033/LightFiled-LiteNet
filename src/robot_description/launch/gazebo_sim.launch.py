import launch
import launch_ros
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # 获取功能包的urdf路径
    urdf_package_path = get_package_share_directory("robot_description")
    # urdf_path = os.path.join(urdf_package_path, "urdf", "my_robot.urdf")
    xacro_path = os.path.join(
        urdf_package_path, "urdf", "light_robot", "lightfieldrobot.urdf.xacro"
    )
    default_world_path = os.path.join(urdf_package_path, "world", "my_world.world")
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
    # 节点启动,提供tf
    action_robot_state_publisher = launch_ros.actions.Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[{"robot_description": robot_description_value}],
    )

    action_launch_gazebo = launch.actions.IncludeLaunchDescription(
        launch.launch_description_sources.PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("gazebo_ros"),
                "launch",
                "gazebo.launch.py",
            )
        ),
        launch_arguments=[("world", default_world_path), ("verbose", "true")],
    )

    action_spawn_entity = launch_ros.actions.Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-topic",
            "robot_description",
            "-entity",
            "my_robot",
        ],
        # spawn_entity.py 的 shebang 是 #!/usr/bin/env python3，会跟着 PATH 走。
        # 本机 conda base 是自动激活的（python3 -> miniconda 的 3.14），那个解释器
        # 既没有 lxml，也 import 不了 Humble 用 3.10 编的 rclpy，于是 spawn 一启动就
        # ModuleNotFoundError 退出 —— Gazebo 能打开，但里面没有车。
        # 这里显式用系统解释器，跟终端里有没有开 conda 无关。
        prefix="/usr/bin/python3 ",
    )

    # 两个轮子是 continuous 关节，robot_state_publisher 只有在收到 /joint_states 之后
    # 才能算出 left/right_wheel_link 的 TF。少了这一步，rviz 的 RobotModel 会在
    # 这两个 link 上打红叉（No transform from [left_wheel_link] to [base_link]）。
    # spawner 会自己等 controller_manager 服务出现，不用手工加延时。
    action_spawner_joint_state = launch_ros.actions.Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
        output="screen",
    )

    # 差速控制器：给 /cmd_vel 走车用，同时发 /odom 和 odom->base_footprint 的 TF。
    # 不需要就走车时把这一段和下面 LaunchDescription 里的引用去掉即可。
    action_spawner_diff_drive = launch_ros.actions.Node(
        package="controller_manager",
        executable="spawner",
        arguments=["mybot_diff_drive_controller"],
        output="screen",
    )

    return launch.LaunchDescription(
        [
            action_declare_arg_mode_path,
            action_robot_state_publisher,
            action_launch_gazebo,
            action_spawn_entity,
            action_spawner_joint_state,
            action_spawner_diff_drive,
        ]
    )
