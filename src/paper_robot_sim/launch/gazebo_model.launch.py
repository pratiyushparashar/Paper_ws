"""
gazebo_model.launch.py (paper_ws) — spawn differential_drive_robot in Gazebo Sim.

Launch arguments:
  world        world file in paper_robot_sim/model/ (or an absolute path)
               default: indoor_world.sdf     slip world: indoor_world_slip.sdf
  gyro_bias_z  gyro z bias_mean in rad/s
               default: 0.0000075 (original robot)   gyro-bias variant: 0.01
  imu_rate     IMU output rate in Hz; default 1000 (= physics step, adopted 2026-10-05 after
               check_imu_integration PASS on session 20261005_045207). Recordings are averaged
               back to 50 Hz offline so accelerations are integrated, not point-sampled; noise
               is scaled so the averaged 50 Hz noise equals the original. imu_rate:=50 gives
               the original point-sampled IMU (start/stop errors up to 0.1 m/s, impacts missed).
  x, y, yaw    spawn pose (m, m, rad); default 0 0 0
               terrain_world.sdf lane i starts at x=0, y=3.8*i (facing +x)

Examples:
  ros2 launch paper_robot_sim gazebo_model.launch.py
  ros2 launch paper_robot_sim gazebo_model.launch.py world:=indoor_world_slip.sdf
  ros2 launch paper_robot_sim gazebo_model.launch.py gyro_bias_z:=0.01
  ros2 launch paper_robot_sim gazebo_model.launch.py world:=terrain_world.sdf y:=15.2
  ros2 launch paper_robot_sim gazebo_model.launch.py imu_rate:=50    # original IMU

Changes vs ws_mobile/mobile_robot: package renamed; gz_args "-r -v4" (Gazebo 10 rejects
"-v -v4"); world and gyro bias selectable via launch arguments (OpaqueFunction, because
xacro is processed in Python and needs the argument values as strings).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro

PACKAGE = 'paper_robot_sim'
ROBOT_NAME = 'differential_drive_robot'   # must match the robot name in robot.xacro


def launch_setup(context, *args, **kwargs):
    share = get_package_share_directory(PACKAGE)

    world = LaunchConfiguration('world').perform(context)
    world_path = world if os.path.isabs(world) else os.path.join(share, 'model', world)
    if not os.path.exists(world_path):
        raise FileNotFoundError(f'World file not found: {world_path}')

    gyro_bias_z = LaunchConfiguration('gyro_bias_z').perform(context)
    float(gyro_bias_z)  # fail early on a non-numeric value
    imu_rate = LaunchConfiguration('imu_rate').perform(context)
    if not 0 < float(imu_rate) <= 1000:
        raise ValueError(f'imu_rate must be in (0, 1000] Hz (physics step 1 ms), got {imu_rate}')
    sx, sy, syaw = (float(LaunchConfiguration(k).perform(context)) for k in ('x', 'y', 'yaw'))

    robot_description = xacro.process_file(
        os.path.join(share, 'model', 'robot.xacro'),
        mappings={'gyro_bias_z': gyro_bias_z, 'imu_rate': imu_rate},
    ).toxml()

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': [f'-r -v4 {world_path}'],
                          'on_exit_shutdown': 'true'}.items(),
    )

    spawn = Node(
        package='ros_gz_sim', executable='create',
        arguments=['-name', ROBOT_NAME, '-topic', 'robot_description',
                   '-x', str(sx), '-y', str(sy), '-z', '0.0', '-Y', str(syaw)],
        output='screen',
    )

    robot_state_publisher = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}],
    )

    bridge = Node(
        package='ros_gz_bridge', executable='parameter_bridge',
        arguments=['--ros-args', '-p',
                   f'config_file:={os.path.join(share, "parameters", "bridge_parameters.yaml")}'],
        output='screen',
    )

    print(f'[paper_robot_sim] world={world_path}  gyro_bias_z={gyro_bias_z}  imu_rate={imu_rate}  '
          f'spawn=({sx}, {sy}, yaw {syaw})')
    return [gazebo, spawn, robot_state_publisher, bridge]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='indoor_world.sdf',
                              description='World file in paper_robot_sim/model/ or absolute path'),
        DeclareLaunchArgument('gyro_bias_z', default_value='0.0000075',
                              description='Gyro z bias_mean (rad/s); 0.01 for the gyro-bias variant'),
        DeclareLaunchArgument('imu_rate', default_value='1000',
                              description='IMU output rate (Hz); averaged to 50 Hz offline. 50 = original'),
        DeclareLaunchArgument('x', default_value='0.0', description='spawn x (m)'),
        DeclareLaunchArgument('y', default_value='0.0', description='spawn y (m)'),
        DeclareLaunchArgument('yaw', default_value='0.0', description='spawn yaw (rad)'),
        OpaqueFunction(function=launch_setup),
    ])
