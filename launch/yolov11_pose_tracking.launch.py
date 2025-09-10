import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    # 获取包目录
    pkg_dir = get_package_share_directory('yolov11_pose_tracking')

    # 构建配置文件路径 - 更可靠的方式
    config_file = os.path.join(pkg_dir, 'config', 'params.yaml')

    # 检查文件是否存在
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at: {config_file}")

    return LaunchDescription([
        Node(
            package='yolov11_pose_tracking',
            executable='yolov11_pose_tracking',
            name='yolov11_pose_node',  # 与params.yaml中的节点名一致
            parameters=[config_file],
            output='screen',
            # 添加重映射(如果需要)
            remappings=[
                ('/camera/color/image_raw', '/camera/color/image_raw'),
                ('/camera/depth/image_raw', '/camera/depth/image_raw')
            ]
        )
    ])