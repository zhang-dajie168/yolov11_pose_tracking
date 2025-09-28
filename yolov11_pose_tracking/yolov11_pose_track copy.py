#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Point32
from geometry_msgs.msg import PolygonStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
from typing import List, Dict, Optional
from collections import deque
import threading
import copy

# 导入RDK YOLO模型和BOTSORT跟踪器
from .YOLO_Pose import YOLO11_Pose
from .BOTSort_rdk import BOTSORT, OSNetReID


class TrackedTarget:
    """跟踪目标信息类"""
    def __init__(self, track_id: int, bbox: List[float], feature: np.ndarray, 
                 height_pixels: float, timestamp: float):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.feature = feature
        self.first_seen_time = timestamp
        self.last_seen_time = timestamp
        self.last_update_time = timestamp
        self.lost_frames = 0
        self.is_recovered = False
        self.is_switched = False  # 新增：标记是否是切换的目标
        self.original_track_id = track_id  # 新增：原始跟踪ID
        self.recovery_time = None  # 新增：目标找回时间
        self.update_paused = False  # 新增：更新暂停标志
    
    def update(self, bbox: List[float], feature: np.ndarray, height_pixels: float, timestamp: float):
        """更新目标信息"""
        self.bbox = bbox
        self.feature = feature
        self.last_seen_time = timestamp
        self.last_update_time = timestamp
        self.lost_frames = 0
        self.is_recovered = False
    
    def mark_lost(self):
        """标记目标丢失"""
        self.lost_frames += 1
        self.update_paused = True  # 丢失后暂停更新
        self.recovery_time = None
    
    def mark_recovered(self, timestamp: float):
        """标记目标找回"""
        self.is_recovered = True
        self.lost_frames = 0
        self.recovery_time = timestamp  # 记录找回时间
        # 不立即恢复更新，等待1秒后再恢复
    
    def switch_to_new_id(self, new_track_id: int):
        """切换到新的跟踪ID"""
        self.is_switched = True
        self.track_id = new_track_id

class Yolov11PoseNode(Node):
    def __init__(self):
        super().__init__('yolov11_pose_node')

        # Camera intrinsics (Orbbec Gemini 335L 640x480)
        self.fx = 367.21
        self.fy = 316.44
        self.cx = 367.20
        self.cy = 244.60

        # Declare and get parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('conf_threshold', 0.3)
        self.declare_parameter('kpt_conf_threshold', 0.25)
        self.declare_parameter('max_processing_fps', 15)
        self.declare_parameter('hands_up_confirm_frames', 3)  # 举手确认帧数
        self.declare_parameter('tracking_protection_time', 5.0)  # 跟踪保护时间（秒）
        self.declare_parameter('reid_similarity_threshold', 0.8)  # ReID相似度阈值
        self.declare_parameter('feature_update_interval', 1.0)  # 特征更新间隔（秒）
        self.declare_parameter('reid_model_path', 'osnet_64x128_nv12.bin')  # ReID模型路径

        # Get parameters
        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.kpt_conf_threshold = self.get_parameter('kpt_conf_threshold').value
        self.max_processing_fps = self.get_parameter('max_processing_fps').value
        self.hands_up_confirm_frames = self.get_parameter('hands_up_confirm_frames').value
        self.tracking_protection_time = self.get_parameter('tracking_protection_time').value
        self.reid_similarity_threshold = self.get_parameter('reid_similarity_threshold').value
        self.feature_update_interval = self.get_parameter('feature_update_interval').value
        reid_model_path = self.get_parameter('reid_model_path').value
        
        self.min_process_interval = 1.0 / self.max_processing_fps
        self.last_process_time = time.time()

        # Load YOLOv11 pose model for RDK
        self.model = YOLO11_Pose(model_path, self.conf_threshold, 0.45)
        
        # Initialize ReID encoder
        self.reid_encoder = None
        try:
            self.reid_encoder = OSNetReID(reid_model_path)
            self.get_logger().info(f"ReID模型加载成功: {reid_model_path}")
        except Exception as e:
            self.get_logger().error(f"ReID模型加载失败: {e}")
        
        # Initialize BOTSORT tracker with ReID enabled
        tracker_args = {
            'track_high_thresh': 0.25,     # 只有置信度高于0.25的检测才会参与主要匹配
            'track_low_thresh': 0.1,       # 置信度在0.1-0.25之间的检测用于补充匹配
            'new_track_thresh': 0.25,      # 只有高分检测才能初始化新轨迹
            'track_buffer': 2,         # 目标丢失后还能在内存中保存30帧，在此期间如果重新出现可以恢复跟踪。
            'match_thresh': 0.68,    # IoU距离小于0.8的才认为是有效匹配
            'fuse_score': False,    # 将检测置信度融入匹配成本计算
            'gmc_method': 'sparseOptFlow',   # 使用稀疏光流进行相机运动补偿，提高在相机移动时的跟踪稳定性
            'proximity_thresh': 0.5,     # 空间距离阈值，用于限制特征匹配的范围
            'appearance_thresh': 0.7,    # 特征相似度阈值，高于0.8才认为是同一人
            'with_reid': False,    # 启用行人重识别功能，提高ID切换的准确性
            'reid_model_path': reid_model_path
        }
        self.tracker = BOTSORT(tracker_args)
        
        # Initialize skeleton connections
        self.skeleton = [
            (16, 14), (14, 12), (17, 15), (15, 13), (12, 13),
            (6, 12), (7, 13), (6, 7), (6, 8), (7, 9),
            (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
            (2, 4), (3, 5), (4, 6), (5, 7)
        ]
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Create subscriptions and publishers
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.detect_pose_pub = self.create_publisher(Image, 'tracks', 10)
        self.keypoint_tracks_pub = self.create_publisher(PolygonStamped, '/keypoint_tracks', 10)
        
        # Tracking and depth related variables
        self.tracked_persons: Dict[int, Dict] = {}

        # 举手检测相关变量
        self.hands_up_history: Dict[int, deque] = {}
        
        # 当前正在跟踪的目标ID
        self.current_tracking_id = None
        
        # 跟踪目标信息存储
        self.tracked_targets: Dict[int, TrackedTarget] = {}
        
        # 新增：目标丢失时间记录
        self.target_lost_time: Optional[float] = None
        
        # 关键点索引定义
        self.KEYPOINT_NAMES = {
            'NOSE': 0,
            'LEFT_EYE': 1, 'RIGHT_EYE': 2,
            'LEFT_EAR': 3, 'RIGHT_EAR': 4,
            'LEFT_SHOULDER': 5, 'RIGHT_SHOULDER': 6,
            'LEFT_ELBOW': 7, 'RIGHT_ELBOW': 8,
            'LEFT_WRIST': 9, 'RIGHT_WRIST': 10,
            'LEFT_HIP': 11, 'RIGHT_HIP': 12,
            'LEFT_KNEE': 13, 'RIGHT_KNEE': 14,
            'LEFT_ANKLE': 15, 'RIGHT_ANKLE': 16
        }

        self.get_logger().info("YOLOv11 Pose Node initialized with ReID recovery")
        self.print_parameters()

    def print_parameters(self):
        """打印所有参数信息"""
        self.get_logger().info("===== 参数配置信息 =====")
        self.get_logger().info(f"置信度阈值: {self.conf_threshold}")
        self.get_logger().info(f"关键点置信度阈值: {self.kpt_conf_threshold}")
        self.get_logger().info(f"最大处理帧率: {self.max_processing_fps}FPS")
        self.get_logger().info(f"举手确认帧数: {self.hands_up_confirm_frames}")
        self.get_logger().info(f"跟踪保护时间: {self.tracking_protection_time}s")
        self.get_logger().info(f"ReID相似度阈值: {self.reid_similarity_threshold}")
        self.get_logger().info(f"特征更新间隔: {self.feature_update_interval}s")
        self.get_logger().info("=========================")

    def extract_feature_from_bbox(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """从边界框提取特征"""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        
        if x2 <= x1 or y2 <= y1:
            return np.zeros(512, dtype=np.float32)
        
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(512, dtype=np.float32)
        
        try:
            if self.reid_encoder is not None:
                feature = self.reid_encoder.extract_feature(crop)
                return feature
            else:
                return np.zeros(512, dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f"Feature extraction failed: {e}")
            return np.zeros(512, dtype=np.float32)
        
    def save_tracked_target(self, track_id: int, bbox: List[float], image: np.ndarray, timestamp: float):
        """保存跟踪目标信息 - 优化版：找回后1秒内不更新，之后恢复正常更新频率"""
        if track_id not in self.tracked_targets:
            # 新目标，直接创建
            feature = self.extract_feature_from_bbox(image, bbox)
            height_pixels = bbox[3] - bbox[1]
            self.tracked_targets[track_id] = TrackedTarget(track_id, bbox, feature, height_pixels, timestamp)
            return
        
        target = self.tracked_targets[track_id]
        
        # 检查是否处于找回后的冷却期
        if target.recovery_time is not None:
            time_since_recovery = timestamp - target.recovery_time
            if time_since_recovery < 1.0:  # 找回后1秒内不更新特征
                # 只更新基础信息，不更新特征
                target.bbox = bbox
                target.height_pixels = bbox[3] - bbox[1]
                target.last_seen_time = timestamp
                target.lost_frames = 0
                target.update_paused = True  # 保持暂停状态
                # self.get_logger().info(f"目标 {track_id} 找回冷却期中，跳过特征更新 (剩余{1.0-time_since_recovery:.1f}s)")
                return
            else:
                # 冷却期结束，恢复正常更新
                target.update_paused = False
                target.recovery_time = None
                self.get_logger().info(f"目标 {track_id} 找回冷却期结束，恢复正常更新")
        
        # 如果更新被暂停（目标丢失状态），跳过特征更新
        if target.update_paused:
            # 只更新基础位置信息，不更新特征
            target.bbox = bbox
            target.height_pixels = bbox[3] - bbox[1]
            target.last_seen_time = timestamp
            target.lost_frames = 0
            self.get_logger().info(f"目标 {track_id} 更新暂停中，只更新位置信息")
            return
        
        # 正常更新频率控制
        time_since_update = timestamp - target.last_update_time
        if time_since_update < self.feature_update_interval:
            # 未达到更新间隔，只更新基础信息
            target.bbox = bbox
            target.height_pixels = bbox[3] - bbox[1]
            target.last_seen_time = timestamp
            target.lost_frames = 0
            target.is_recovered = False
            return
        
        # 达到更新间隔，进行完整更新（包括特征）
        feature = self.extract_feature_from_bbox(image, bbox)
        height_pixels = bbox[3] - bbox[1]
        target.update(bbox, feature, height_pixels, timestamp)

    def try_recover_lost_target(self, current_tracks: List[Dict], image: np.ndarray, timestamp: float) -> Optional[int]:
        """立即尝试找回丢失的跟踪目标 - 严格版"""
        if self.current_tracking_id is None or self.current_tracking_id not in self.tracked_targets:
            return None
        
        target = self.tracked_targets[self.current_tracking_id]
        
        # 立即记录丢失时间（不再等待帧数）
        if self.target_lost_time is None:
            self.target_lost_time = timestamp
            self.get_logger().info(f"目标 {self.current_tracking_id} 丢失，开始立即ReID匹配找回")
        
        # 考虑所有当前检测到的目标作为候选
        candidate_tracks = []
        for track in current_tracks:
            track_id = track['track_id']
            
            # 排除当前正在跟踪的目标（除了可能是原始目标重新出现）
            is_currently_tracked = (
                track_id in self.tracked_persons and 
                self.tracked_persons[track_id]['is_tracking'] and
                track_id != self.current_tracking_id  # 原始目标重新出现不算当前跟踪
            )
            
            if not is_currently_tracked:
                candidate_tracks.append(track)
        
        if not candidate_tracks:
            self.get_logger().info(f"目标 {self.current_tracking_id} 找回: 当前帧无候选目标")
            return None
        
        # 对所有候选目标进行严格的ReID匹配
        best_match_id = None
        best_similarity = 0.0
        
        for track in candidate_tracks:
            track_id = track['track_id']
            bbox = track['bbox']
            
            candidate_feature = self.extract_feature_from_bbox(image, bbox)
            
            if candidate_feature is not None and np.any(candidate_feature):
                similarity = np.dot(target.feature, candidate_feature) / (
                    np.linalg.norm(target.feature) * np.linalg.norm(candidate_feature) + 1e-8
                )
                
                # 严格阈值：必须达到0.8以上
                if similarity >= 0.8 and similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = track_id
        
        if best_match_id is not None:
            self.get_logger().info(
                f"目标 {self.current_tracking_id} ReID找回成功! 匹配ID: {best_match_id}, 相似度: {best_similarity:.3f}"
            )
            
            # 更新目标信息（使用优化后的save_tracked_target）
            target_bbox = next(t['bbox'] for t in candidate_tracks if t['track_id'] == best_match_id)
            
            # 标记目标为找回状态
            target.mark_recovered(timestamp)
            
            # 保存目标信息（会进入找回冷却期）
            self.save_tracked_target(self.current_tracking_id, target_bbox, image, timestamp)
            
            # 关键修改：无论找回的是否是原来的ID，都要更新跟踪状态
            recovered_id = best_match_id
            
            # 标记为切换状态（如果ID不同）
            if best_match_id != self.current_tracking_id:
                if best_match_id in self.tracked_targets:
                    self.tracked_targets[best_match_id].switch_to_new_id(best_match_id)
                    self.tracked_targets[best_match_id].original_track_id = self.current_tracking_id
                    # 新ID的目标也标记为找回状态
                    self.tracked_targets[best_match_id].mark_recovered(timestamp)
            
            # 无论找回的是否是原来的ID，都要重置跟踪状态
            self.target_lost_time = None
            
            # 重要：更新跟踪状态，确保可视化正确
            if recovered_id in self.tracked_persons:
                self.tracked_persons[recovered_id]['is_tracking'] = True
                self.tracked_persons[recovered_id]['tracking_start_time'] = timestamp
                self.tracked_persons[recovered_id]['last_seen_time'] = timestamp
            
            return recovered_id
        
        self.get_logger().warning(f"目标 {self.current_tracking_id} ReID找回失败: 无匹配目标达到阈值")
        return None

    def _verify_target_with_reid(self, target: TrackedTarget, track: Dict, image: np.ndarray, timestamp: float) -> Optional[int]:
        """使用ReID验证目标身份"""
        track_id = track['track_id']
        bbox = track['bbox']
        
        candidate_feature = self.extract_feature_from_bbox(image, bbox)
        
        if candidate_feature is not None and np.any(candidate_feature):
            similarity = np.dot(target.feature, candidate_feature) / (
                np.linalg.norm(target.feature) * np.linalg.norm(candidate_feature) + 1e-8
            )
            
            if similarity >= 0.8:  # 严格阈值
                self.get_logger().info(f"ReID验证成功: ID {track_id}, 相似度: {similarity:.3f}")
                
                # 标记目标为找回状态
                target.mark_recovered(timestamp)
                
                # 更新目标信息（使用优化后的save_tracked_target）
                self.save_tracked_target(target.track_id, bbox, image, timestamp)
                
                self.target_lost_time = None
                
                # 关键修改：确保跟踪状态正确更新
                if track_id in self.tracked_persons:
                    self.tracked_persons[track_id]['is_tracking'] = True
                    self.tracked_persons[track_id]['last_seen_time'] = timestamp
                    # 如果是原来的ID重新出现，保持原来的tracking_start_time
                    if track_id == target.track_id:
                        # 保持原来的开始时间，不重置
                        pass
                    else:
                        self.tracked_persons[track_id]['tracking_start_time'] = timestamp
                
                return track_id
            else:
                self.get_logger().warning(f"ReID验证失败: ID {track_id}, 相似度: {similarity:.3f}")
        
        return None


    def image_callback(self, msg):
        current_time = time.time()
        if current_time - self.last_process_time < self.min_process_interval:
            return
        
        # total_start_time = time.time()
        self.last_process_time = current_time

        try:
            # 图像转换
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # YOLO推理
            input_tensor = self.model.bgr2nv12(cv_image)
            outputs = self.model.c2numpy(self.model.forward(input_tensor))
            
            # 后处理
            ids, scores, bboxes, kpts_xy, kpts_score = self.model.postProcess(outputs)

            # 准备检测结果用于跟踪
            detections = []
            person_kpts_xy = []
            person_kpts_score = []

            for i in range(len(ids)):
                if ids[i] == 0:
                    x1, y1, x2, y2 = bboxes[i]
                    w, h = x2 - x1, y2 - y1
                    detections.append([x1, y1, w, h, scores[i], 0])
                    person_kpts_xy.append(kpts_xy[i])
                    person_kpts_score.append(kpts_score[i])

            # 跟踪
            tracking_results = self.tracker.update(detections, cv_image, person_kpts_xy, person_kpts_score)

            # 处理跟踪结果
            tracks = []
            for result in tracking_results:
                x, y, w, h, track_id, score, cls, keypoints, keypoints_conf = result
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                
                track_data = {
                    'track_id': int(track_id),
                    'bbox': [x1, y1, x2, y2],
                    'conf': float(score),
                    'keypoints': keypoints,
                    'keypoints_conf': keypoints_conf
                }
                tracks.append(track_data)

            # 更新跟踪状态
            current_track_ids = set()
            for track in tracks:
                track_id = track['track_id']
                current_track_ids.add(track_id)
                
                if track_id not in self.tracked_persons:
                    self.tracked_persons[track_id] = {
                        'is_tracking': False,
                        'tracking_start_time': 0.0,
                        'last_hands_up_time': 0.0,
                        'first_seen_time': current_time,
                        'last_seen_time': current_time
                    }
                    self.hands_up_history[track_id] = deque(maxlen=self.hands_up_confirm_frames)
                else:
                    self.tracked_persons[track_id]['last_seen_time'] = current_time

                person = self.tracked_persons[track_id]
                hands_up = self.is_hands_up(track)
                
                self.hands_up_history[track_id].append(hands_up)
                hands_up_confirmed = sum(self.hands_up_history[track_id]) >= self.hands_up_confirm_frames
                
                if self.current_tracking_id is None:
                    in_cooldown_period = (current_time - person['last_hands_up_time'] < self.tracking_protection_time)
                    
                    if not in_cooldown_period and hands_up_confirmed:
                        self.current_tracking_id = track_id
                        person['is_tracking'] = True
                        person['tracking_start_time'] = current_time
                        person['last_hands_up_time'] = current_time
                        self.hands_up_history[track_id].clear()
                        self.save_tracked_target(track_id, track['bbox'], cv_image, current_time)
                        self.target_lost_time = None
                        self.get_logger().info(f"开始跟踪 ID: {track_id} (举手确认)")
                
                elif self.current_tracking_id == track_id:
                    in_protection_period = (current_time - person['tracking_start_time'] < self.tracking_protection_time)
                    
                    if not in_protection_period and hands_up_confirmed:
                        person['is_tracking'] = False
                        person['last_hands_up_time'] = current_time
                        self.hands_up_history[track_id].clear()
                        self.current_tracking_id = None
                        self.target_lost_time = None
                        self.get_logger().info(f"停止跟踪 ID: {track_id}")
                    else:
                        self.save_tracked_target(track_id, track['bbox'], cv_image, current_time)

            # 处理丢失目标 - 关键修改部分
            if self.current_tracking_id is not None and self.current_tracking_id not in current_track_ids:
                if self.current_tracking_id in self.tracked_targets:
                    self.tracked_targets[self.current_tracking_id].mark_lost()
                    
                    # 立即标记目标丢失状态
                    if self.current_tracking_id in self.tracked_persons:
                        self.tracked_persons[self.current_tracking_id]['is_tracking'] = False
                    
                    if self.target_lost_time is None:
                        self.target_lost_time = current_time
                        self.get_logger().warning(f"目标 {self.current_tracking_id} 丢失，立即启动ReID找回")
                    
                    # 立即尝试ReID找回
                    recovered_id = self.try_recover_lost_target(tracks, cv_image, current_time)
                    if recovered_id is not None:
                        # ReID找回成功，切换到新ID
                        self.current_tracking_id = recovered_id
                        if recovered_id in self.tracked_persons:
                            self.tracked_persons[recovered_id]['is_tracking'] = True
                            self.tracked_persons[recovered_id]['tracking_start_time'] = current_time
                            self.get_logger().info(f"ReID找回成功，切换到新ID: {recovered_id}")
                    else:
                        # ReID找回失败，保持目标丢失状态
                        self.get_logger().warning(f"目标 {self.current_tracking_id} ReID找回失败，保持丢失状态")
            

            elif self.current_tracking_id is not None and self.current_tracking_id in current_track_ids:
                # 检查是否是目标丢失后重新出现的（需要ReID验证）
                if self.target_lost_time is not None:
                    self.get_logger().info(f"目标 {self.current_tracking_id} 重新出现，进行ReID验证")
                    track = next(t for t in tracks if t['track_id'] == self.current_tracking_id)
                    verified_id = self._verify_target_with_reid(
                        self.tracked_targets[self.current_tracking_id], track, cv_image, current_time
                    )
                    
                    if verified_id is not None:
                        # ReID验证成功，继续跟踪
                        self.target_lost_time = None
                        self.get_logger().info(f"目标 {self.current_tracking_id} ReID验证成功，继续跟踪")
                        
                        # 关键修改：确保跟踪状态正确设置
                        if verified_id in self.tracked_persons:
                            self.tracked_persons[verified_id]['is_tracking'] = True
                            self.tracked_persons[verified_id]['last_seen_time'] = current_time

            # 清理长时间未出现的跟踪目标
            self.cleanup_old_tracks(current_time, current_track_ids)

            # 可视化并发布结果
            self.visualize_results(cv_image, tracks)
            self.publish_tracked_keypoints(tracks, msg.header)

            # 只在有订阅者时才进行可视化发布
            if self.detect_pose_pub.get_subscription_count() > 0:
                detect_pose_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
                detect_pose_msg.header = msg.header
                self.detect_pose_pub.publish(detect_pose_msg)

        except Exception as e:
            self.get_logger().error(f"Image processing error: {str(e)}")

    def cleanup_old_tracks(self, current_time, current_track_ids):
        """清理长时间未出现的跟踪目标"""
        max_track_age = 5.0
        
        for track_id in list(self.tracked_persons.keys()):
            if track_id == self.current_tracking_id:
                continue
                
            if track_id not in current_track_ids:
                last_seen = self.tracked_persons[track_id]['last_seen_time']
                if current_time - last_seen > max_track_age:
                    del self.tracked_persons[track_id]
                    if track_id in self.hands_up_history:
                        del self.hands_up_history[track_id]
                    if track_id in self.tracked_targets:
                        # 如果是被切换的目标，检查是否需要恢复原始ID
                        target = self.tracked_targets[track_id]
                        if target.is_switched and target.original_track_id not in self.tracked_targets:
                            original_id = target.original_track_id
                            self.tracked_targets[original_id] = copy.deepcopy(target)
                            self.tracked_targets[original_id].track_id = original_id
                            self.tracked_targets[original_id].is_switched = False
                        
                        del self.tracked_targets[track_id]
                    # self.get_logger().info(f"清理长时间未出现的跟踪目标 ID: {track_id}")

    def is_hands_up(self, track: Dict) -> bool:
        """举手检测"""
        keypoints = track['keypoints']
        keypoints_conf = track['keypoints_conf']

        def has_valid_keypoint(index):
            return (index < len(keypoints) and 
                    not np.isnan(keypoints[index]).any() and 
                    not (keypoints[index][0] == 0 and keypoints[index][1] == 0) and
                    keypoints_conf[index] >= self.kpt_conf_threshold)

        if not (has_valid_keypoint(self.KEYPOINT_NAMES['LEFT_SHOULDER']) and 
                has_valid_keypoint(self.KEYPOINT_NAMES['RIGHT_SHOULDER']) and
                has_valid_keypoint(self.KEYPOINT_NAMES['NOSE'])):
            return False

        left_shoulder = keypoints[self.KEYPOINT_NAMES['LEFT_SHOULDER']]
        right_shoulder = keypoints[self.KEYPOINT_NAMES['RIGHT_SHOULDER']]
        nose = keypoints[self.KEYPOINT_NAMES['NOSE']]

        left_hand_up = False
        if (has_valid_keypoint(self.KEYPOINT_NAMES['LEFT_WRIST']) and 
            has_valid_keypoint(self.KEYPOINT_NAMES['LEFT_ELBOW'])):
            left_wrist = keypoints[self.KEYPOINT_NAMES['LEFT_WRIST']]
            # left_elbow = keypoints[self.KEYPOINT_NAMES['LEFT_ELBOW']]
            left_hand_up = (left_wrist[1] < left_shoulder[1] and
                           left_wrist[1] < nose[1] and
                           abs(left_wrist[0] - left_shoulder[0]) < 80)

        right_hand_up = False
        if (has_valid_keypoint(self.KEYPOINT_NAMES['RIGHT_WRIST']) and 
            has_valid_keypoint(self.KEYPOINT_NAMES['RIGHT_ELBOW'])):
            right_wrist = keypoints[self.KEYPOINT_NAMES['RIGHT_WRIST']]
            # right_elbow = keypoints[self.KEYPOINT_NAMES['RIGHT_ELBOW']]
            right_hand_up = (right_wrist[1] < right_shoulder[1] and
                            right_wrist[1] < nose[1] and
                            abs(right_wrist[0] - right_shoulder[0]) < 80)

        return left_hand_up or right_hand_up
        
    def visualize_results(self, image: np.ndarray, tracks: List[Dict]):
        """简化版可视化跟踪结果"""
        display_image = image.copy()
        
        for track in tracks:
            track_id = track['track_id']
            x1, y1, x2, y2 = track['bbox']
            confidence = track['conf']  # 获取置信度

            # 确保只有一个目标被跟踪
            is_tracking = (track_id == self.current_tracking_id and 
                        track_id in self.tracked_persons and 
                        self.tracked_persons[track_id]['is_tracking'])
            
            if is_tracking:
                tracking_time = time.time() - self.tracked_persons[track_id]['tracking_start_time']
                in_protection_period = tracking_time < self.tracking_protection_time
            else:
                in_protection_period = False

            # 设置颜色（统一使用框的颜色）
            if is_tracking:
                if in_protection_period:
                    color = (255, 165, 0)  # 橙色 - 保护期内
                else:
                    color = (255, 0, 0)    # 红色 - 跟踪中
            else:
                color = (0, 255, 0)        # 绿色 - 未跟踪

            # 绘制边界框
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, 2)

            # 简化的显示信息
            if track_id in self.tracked_persons:
                person = self.tracked_persons[track_id]
                if is_tracking:
                    if in_protection_period:
                        protection_left = self.tracking_protection_time - tracking_time
                        label = f"ID:{track_id} C:{confidence:.2f} P:{protection_left:.0f}s"
                    else:
                        label = f"ID:{track_id} C:{confidence:.2f} T:{tracking_time:.0f}s"
                else:
                    confirm_count = sum(self.hands_up_history.get(track_id, []))
                    label = f"ID:{track_id} C:{confidence:.2f} R({confirm_count}/{self.hands_up_confirm_frames})"
            else:
                label = f"ID:{track_id} C:{confidence:.2f}"

            # 简化的文本绘制（去掉白边，直接使用框的颜色）
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            
            # 获取文本尺寸
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # 计算文本位置（框上方）
            text_x = x1
            text_y = max(y1 - 5, text_height + 5)  # 确保不超出图像顶部
                
            # 直接绘制文本（使用框的相同颜色，去掉白边）
            cv2.putText(display_image, label, 
                    (text_x, text_y), 
                    font, font_scale, color, thickness)

            # 绘制关键点（简化：只画可见的点）
            keypoints = track['keypoints']
            keypoints_conf = track['keypoints_conf']
            
            for i, kp in enumerate(keypoints):
                if (i < len(keypoints_conf) and 
                    keypoints_conf[i] >= self.kpt_conf_threshold and
                    not np.isnan(kp).any() and 
                    not (kp[0] == 0 and kp[1] == 0)):
                    x, y = kp.astype(int)
                    cv2.circle(display_image, (x, y), 2, (0, 0, 255), -1)

            # 绘制骨架（简化：只画高置信度的连接）
            for connection in self.skeleton:
                idx1, idx2 = connection
                idx1 -= 1
                idx2 -= 1
                
                if (idx1 < len(keypoints) and idx2 < len(keypoints) and
                    idx1 < len(keypoints_conf) and idx2 < len(keypoints_conf) and
                    keypoints_conf[idx1] >= self.kpt_conf_threshold and
                    keypoints_conf[idx2] >= self.kpt_conf_threshold and
                    not np.isnan(keypoints[idx1]).any() and 
                    not np.isnan(keypoints[idx2]).any() and
                    not (keypoints[idx1][0] == 0 and keypoints[idx1][1] == 0) and
                    not (keypoints[idx2][0] == 0 and keypoints[idx2][1] == 0)):
                    
                    pt1 = keypoints[idx1].astype(int)
                    pt2 = keypoints[idx2].astype(int)
                    cv2.line(display_image, pt1, pt2, (0, 255, 255), 1)

        image[:] = display_image[:]
        
    def publish_tracked_keypoints(self, tracks: List[Dict], header):
        """发布边界框和肩部关键点坐标和置信度"""
        # 检查当前是否有跟踪目标
        has_tracking_target = False
        current_tracking_id = self.current_tracking_id
        
        for track in tracks:
            track_id = track['track_id']
            # 检查是否是当前跟踪的目标
            is_tracking_target = (
                (current_tracking_id is not None and track_id == current_tracking_id) or
                (track_id in self.tracked_persons and self.tracked_persons[track_id]['is_tracking'])
            )
            
            if is_tracking_target:
                has_tracking_target = True
                x1, y1, x2, y2 = track['bbox']
                keypoints = track['keypoints']
                keypoints_conf = track['keypoints_conf']
                
                polygon_msg = PolygonStamped()
                polygon_msg.header = header
                polygon_msg.header.frame_id = "camera_link"
                
                # 第一个点存储状态信息和关键点数量
                # x=track_id, y=状态(1正常/0丢失), z=关键点数量(这里固定为2，表示只发布两个肩部点)
                points = [
                    Point32(x=float(track_id), y=1.0, z=2.0),  # 状态信息：z=2表示两个关键点
                    Point32(x=float(x1), y=float(y1), z=0.0),   # 边界框左上角
                    Point32(x=float(x2), y=float(y2), z=0.0),   # 边界框右下角
                ]
                
                # 只添加左肩和右肩两个关键点
                shoulder_indices = [
                    self.KEYPOINT_NAMES['LEFT_SHOULDER'],
                    self.KEYPOINT_NAMES['RIGHT_SHOULDER']
                ]
                
                # shoulder_names = ['LEFT_SHOULDER', 'RIGHT_SHOULDER']
                
                for i, kp_index in enumerate(shoulder_indices):
                    if kp_index < len(keypoints):
                        kp = keypoints[kp_index]
                        conf = keypoints_conf[kp_index] if kp_index < len(keypoints_conf) else 0.0
                        
                        if not np.isnan(kp).any() and not (kp[0] == 0 and kp[1] == 0):
                            # 关键点坐标 (x, y坐标)
                            points.append(Point32(x=float(kp[0]), y=float(kp[1]), z=float(conf)))
                        
                        else:
                            # 无效关键点
                            points.append(Point32(x=0.0, y=0.0, z=0.0))
                    else:
                        # 关键点索引超出范围
                        points.append(Point32(x=0.0, y=0.0, z=0.0))

                polygon_msg.polygon.points = points
                self.keypoint_tracks_pub.publish(polygon_msg)
                
        
        # 如果当前应该有跟踪目标但目标丢失了
        if current_tracking_id is not None and not has_tracking_target:
            polygon_msg = PolygonStamped()
            polygon_msg.header = header
            polygon_msg.header.frame_id = "camera_link"
            
            # 发布目标丢失状态
            points = [
                Point32(x=float(current_tracking_id), y=0.0, z=0.0),  # 状态点：y=0表示目标丢失
                Point32(x=0.0, y=0.0, z=0.0),  # 无效边界框坐标
                Point32(x=0.0, y=0.0, z=0.0),  # 无效边界框坐标
            ]
            
            polygon_msg.polygon.points = points
            self.keypoint_tracks_pub.publish(polygon_msg)
            self.get_logger().info(f"发布目标丢失状态: ID {current_tracking_id}")
        
        # 处理跟踪被取消的情况
        elif current_tracking_id is None:
            polygon_msg = PolygonStamped()
            polygon_msg.header = header
            polygon_msg.header.frame_id = "camera_link"
            
            points = [
                Point32(x=0.0, y=0.0, z=0.0),  # 无跟踪目标
                Point32(x=0.0, y=0.0, z=0.0),
                Point32(x=0.0, y=0.0, z=0.0),
            ]
            
            polygon_msg.polygon.points = points
            self.keypoint_tracks_pub.publish(polygon_msg)


    def print_tracking_info(self, tracks: List[Dict]):
        """打印跟踪信息"""
        if self.current_tracking_id is not None:
            if self.current_tracking_id in self.tracked_persons:
                person = self.tracked_persons[self.current_tracking_id]
                tracking_time = time.time() - person['tracking_start_time']
                if tracking_time < self.tracking_protection_time:
                    protection_left = self.tracking_protection_time - tracking_time
                    self.get_logger().info(f"当前跟踪目标: ID {self.current_tracking_id}, 保护期剩余: {protection_left:.1f}s")
                else:
                    self.get_logger().info(f"当前跟踪目标: ID {self.current_tracking_id}, 已跟踪: {tracking_time:.1f}s")
            else:
                self.get_logger().info(f"当前跟踪目标: ID {self.current_tracking_id} (状态未知)")
        else:
            ready_count = sum(1 for track in tracks if track['track_id'] in self.tracked_persons and 
                            not self.tracked_persons[track['track_id']]['is_tracking'])
            self.get_logger().info(f"无跟踪目标，{ready_count}个目标可跟踪")


def main(args=None):
    rclpy.init(args=args)
    node = Yolov11PoseNode()
    executor = MultiThreadedExecutor()
    rclpy.spin(node, executor)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
