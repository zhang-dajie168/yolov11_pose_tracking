
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


当前代码中的目标丢失找回逻辑主要包含以下几个核心部分：
1. 丢失检测机制

    标记丢失：当当前跟踪目标(current_tracking_id)不在当前帧的检测结果中时，标记为目标丢失

    记录丢失时间：第一次检测到丢失时记录target_lost_time时间戳

    丢失帧计数：通过TrackedTarget.lost_frames记录连续丢失的帧数

2. 找回条件判断

找回操作仅在满足以下条件时触发：

    当前有正在跟踪的目标(current_tracking_id不为None)

    目标已丢失且丢失帧数达到阈值(max_lost_frames_for_recovery)

    目标在跟踪目标字典中存在对应的TrackedTarget记录

3. 候选目标筛选

从当前帧的检测结果中筛选合适的候选目标，条件包括：

    非当前跟踪目标：不是已经被标记为正在跟踪的目标

    时间相关性：目标出现时间在丢失时间之后（新出现或重新出现）

    避免旧目标：只考虑3秒内出现的新目标或原始目标本身

4. 特征匹配找回

    特征提取：使用ReID模型从候选目标的边界框提取特征向量

    相似度计算：计算候选目标特征与丢失目标特征的余弦相似度

    优先级设置：原始目标重新出现时优先选择（相似度设为0.9以上）

    阈值判断：相似度超过reid_similarity_threshold且为最佳匹配时选择

5. 找回后的处理

    状态更新：重置丢失帧数，标记为已找回状态

    ID处理：

        如果是原始目标重新出现，直接恢复原ID

        如果是新目标匹配，将新目标标记为切换状态，并记录原始ID

    时间重置：清除target_lost_time记录

6. 清理机制

    长时间未出现清理：5秒未出现的非当前跟踪目标会被清理

    切换目标处理：清理切换目标时，如果需要会恢复原始目标的跟踪信息

关键参数配置

    reid_similarity_threshold: ReID相似度阈值(0.7)

    max_lost_frames_for_recovery: 最大丢失帧数(5帧)

    feature_update_interval: 特征更新间隔(5秒)

这个逻辑结合了时间判断、外观特征匹配和优先级处理，能够在目标短暂消失或外观变化时有效找回跟踪目标。

