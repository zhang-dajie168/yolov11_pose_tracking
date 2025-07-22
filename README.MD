
# 1.build
colcon build --packages-select yolov11_pose_tracking

# 2. run
 source install/setup.bash
 ros2 launch yolov11_pose_tracking yolov11_pose_tracking.launch.py 
 
 
# 3. echo

ros2 topic echo /person_positions


# 4.result

header:
  stamp:
    sec: 1753167759
    nanosec: 513306880
  frame_id: camera_link
point:
  x: 0.49863838130769866
  y: -0.017254455821008704
  z: 0.975



