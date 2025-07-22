# launch/yolov11_pose_tracking.launch.py
import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('yolov11_pose_tracking'),
        'config',
        'params.yaml'
    )

    return LaunchDescription([
        Node(
            package='yolov11_pose_tracking',
            executable='yolov11_pose_tracking',
            name='yolov11_pose_tracking',
            parameters=[config],
            output='screen'
        )
    ])