import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
import xacro
def generate_launch_description():

    # this name has to match the robot name in the Xacro file
    robotXacroName ='differential_drive_robot'

    # this is the name of our package, at the same time this is the name of the
    # folder that will be used to define the paths
    namePackage = 'paper_robot_sim'

    # this is a relative path to the xacro file defining the model
    modelFileRelativePath = 'model/robot.xacro'

    # this is the absolute path to the model
    pathModelFile = os.path.join(get_package_share_directory(namePackage),modelFileRelativePath)

    robotDescription = xacro.process_file(pathModelFile).toxml()

    gazebo_rosPackageLaunch = PythonLaunchDescriptionSource(
    os.path.join(get_package_share_directory('ros_gz_sim'),'launch','gz_sim.launch.py'))

    worldFileRelativePath = 'model/indoor_world.sdf'
    pathWorldFile = os.path.join(get_package_share_directory(namePackage), worldFileRelativePath)

    gazeboLaunch=IncludeLaunchDescription(gazebo_rosPackageLaunch, launch_arguments={'gz_args':[f'-r -v4 {pathWorldFile}'], 'on_exit_shutdown':'true'}.items())
    
    spawnModelNodeGazebo = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', robotXacroName,
            '-topic', 'robot_description'
        ],
        output='screen',
    )
    
    nodeRobotStatePublisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robotDescription,
        'use_sim_time': True}]
    )

    bridge_params = os.path.join(
    get_package_share_directory(namePackage),
    'parameters',
    'bridge_parameters.yaml'
    )
    
    start_gazebo_ros_bridge_cmd = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
        '--ros-args',
        '-p',
        f'config_file:={bridge_params}',
    ],
    output='screen',
    )
    launchDescriptionObject = LaunchDescription()
    launchDescriptionObject.add_action(gazeboLaunch)
    launchDescriptionObject.add_action(spawnModelNodeGazebo)
    launchDescriptionObject.add_action(nodeRobotStatePublisher)
    launchDescriptionObject.add_action(start_gazebo_ros_bridge_cmd)
    return launchDescriptionObject
