from setuptools import setup

package_name = 'yolov11_pose_tracking'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', ['config/params.yaml']),
        ('share/' + package_name + '/launch', ['launch/yolov11_pose_tracking.launch.py'])
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
