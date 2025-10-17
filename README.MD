
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


# 跟踪找回逻辑：

目标跟踪中
    ↓ (目标消失)
目标丢失 → 启动主动找回
    ↓ (搜索候选目标)
ReID匹配 → 相似度≥阈值 → 匹配成功 → 身份切换 → 恢复跟踪
    ↓ (匹配失败)          ↓ (验证失败)
保持丢失状态 ←────── 目标重现 → ReID验证 → 验证成功 → 恢复跟踪
