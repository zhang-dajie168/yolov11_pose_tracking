from setuptools import setup
import os
from glob import glob

package_name = 'yolov11_pose_tracking'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 确保正确安装所有配置文件
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
        # 如果需要模型文件也一起安装
        (os.path.join('share', package_name, 'models'),
         glob('models/*.pt')),
        # 安装消息文件
        (os.path.join('share', package_name, 'msg'),
         glob('msg/*.msg')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your.email@example.com',
    description='ROS2 node for YOLOv11 pose tracking with hand-raising detection',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolov11_pose_tracking = yolov11_pose_tracking.yolov11_pose_track:main',
        ],
    },
)