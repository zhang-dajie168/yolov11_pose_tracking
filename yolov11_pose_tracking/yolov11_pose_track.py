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

class DepthFilter:
    def __init__(self, depth_window_size=5, depth_threshold=0.5):
        self.depth_window_size = depth_window_size
        self.depth_threshold = depth_threshold
        self.depth_window = deque(maxlen=depth_window_size)

    def add_depth(self, depth):
        self.depth_window.append(depth)

    def get_filtered_depth(self):
        if len(self.depth_window) == 0:
            return 0.0
        return np.median(self.depth_window) if len(self.depth_window) > 1 else self.depth_window[-1]

class TrackedTarget:
    """跟踪目标信息类"""
    def __init__(self, track_id: int, bbox: List[float], feature: np.ndarray, 
                 height_pixels: float, timestamp: float):
        self.track_id = track_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.feature = feature
        self.height_pixels = height_pixels
        self.first_seen_time = timestamp
        self.last_seen_time = timestamp
        self.last_update_time = timestamp
        self.lost_frames = 0
        self.is_recovered = False
        self.is_switched = False  # 新增：标记是否是切换的目标
        self.original_track_id = track_id  # 新增：原始跟踪ID
    
    def update(self, bbox: List[float], feature: np.ndarray, height_pixels: float, timestamp: float):
        """更新目标信息"""
        self.bbox = bbox
        self.feature = feature
        self.height_pixels = height_pixels
        self.last_seen_time = timestamp
        self.last_update_time = timestamp
        self.lost_frames = 0
        self.is_recovered = False
    
    def mark_lost(self):
        """标记目标丢失"""
        self.lost_frames += 1
    
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
        self.declare_parameter('real_person_height', 1.7)
        self.declare_parameter('max_processing_fps', 15)
        self.declare_parameter('hands_up_confirm_frames', 3)  # 举手确认帧数
        self.declare_parameter('tracking_protection_time', 10.0)  # 跟踪保护时间（秒）
        self.declare_parameter('reid_similarity_threshold', 0.7)  # ReID相似度阈值
        self.declare_parameter('max_lost_frames_for_recovery', 5)  # 最大丢失帧数用于找回
        self.declare_parameter('feature_update_interval', 5.0)  # 特征更新间隔（秒）
        self.declare_parameter('reid_model_path', 'osnet_64x128_nv12.bin')  # ReID模型路径

        # Get parameters
        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.kpt_conf_threshold = self.get_parameter('kpt_conf_threshold').value
        self.real_person_height = self.get_parameter('real_person_height').value
        self.max_processing_fps = self.get_parameter('max_processing_fps').value
        self.hands_up_confirm_frames = self.get_parameter('hands_up_confirm_frames').value
        self.tracking_protection_time = self.get_parameter('tracking_protection_time').value
        self.reid_similarity_threshold = self.get_parameter('reid_similarity_threshold').value
        self.max_lost_frames_for_recovery = self.get_parameter('max_lost_frames_for_recovery').value
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
            'track_high_thresh': 0.25,
            'track_low_thresh': 0.1,
            'new_track_thresh': 0.25,
            'track_buffer': 30,
            'match_thresh': 0.8,
            'fuse_score': True,
            'gmc_method': 'sparseOptFlow',
            'proximity_thresh': 0.5,
            'appearance_thresh': 0.7,
            'with_reid': False,
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
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.detect_pose_pub = self.create_publisher(Image, 'tracks', 10)
        self.person_point_pub = self.create_publisher(PointStamped, '/person_positions', 10)
        self.keypoint_tracks_pub = self.create_publisher(PolygonStamped, '/keypoint_tracks', 10)
        
        # Tracking and depth related variables
        self.tracked_persons: Dict[int, Dict] = {}
        self.depth_image = None
        self.depth_lock = threading.Lock()
        self.depth_filters: Dict[int, DepthFilter] = {}
        
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
        self.get_logger().info(f"模型路径: {self.get_parameter('model_path').value}")
        self.get_logger().info(f"置信度阈值: {self.conf_threshold}")
        self.get_logger().info(f"关键点置信度阈值: {self.kpt_conf_threshold}")
        self.get_logger().info(f"真实人身高: {self.real_person_height}m")
        self.get_logger().info(f"最大处理帧率: {self.max_processing_fps}FPS")
        self.get_logger().info(f"举手确认帧数: {self.hands_up_confirm_frames}")
        self.get_logger().info(f"跟踪保护时间: {self.tracking_protection_time}s")
        self.get_logger().info(f"ReID相似度阈值: {self.reid_similarity_threshold}")
        self.get_logger().info(f"最大丢失帧数用于找回: {self.max_lost_frames_for_recovery}")
        self.get_logger().info(f"特征更新间隔: {self.feature_update_interval}s")
        self.get_logger().info(f"ReID模型路径: {self.get_parameter('reid_model_path').value}")
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
        """保存跟踪目标信息"""
        if track_id in self.tracked_targets:
            target = self.tracked_targets[track_id]
            time_since_update = timestamp - target.last_update_time
            if time_since_update < self.feature_update_interval:
                target.bbox = bbox
                target.height_pixels = bbox[3] - bbox[1]
                target.last_seen_time = timestamp
                target.lost_frames = 0
                target.is_recovered = False
                return
        
        feature = self.extract_feature_from_bbox(image, bbox)
        height_pixels = bbox[3] - bbox[1]
        
        if track_id in self.tracked_targets:
            self.tracked_targets[track_id].update(bbox, feature, height_pixels, timestamp)
        else:
            self.tracked_targets[track_id] = TrackedTarget(track_id, bbox, feature, height_pixels, timestamp)

    def try_recover_lost_target(self, current_tracks: List[Dict], image: np.ndarray, timestamp: float) -> Optional[int]:
        """尝试找回丢失的跟踪目标"""
        if self.current_tracking_id is None or self.current_tracking_id not in self.tracked_targets:
            return None
        
        target = self.tracked_targets[self.current_tracking_id]
        
        if target.lost_frames < self.max_lost_frames_for_recovery:
            return None
        
        # 如果是第一次检测到丢失，记录丢失时间
        if self.target_lost_time is None:
            self.target_lost_time = timestamp
            self.get_logger().info(f"目标 {self.current_tracking_id} 丢失时间记录: {timestamp}")
        
        # 考虑所有当前检测到但不在跟踪状态的目标
        candidate_tracks = []
        for track in current_tracks:
            track_id = track['track_id']
            
            # 条件1：不是当前正在跟踪的目标
            is_currently_tracked = (track_id in self.tracked_persons and 
                                self.tracked_persons[track_id]['is_tracking'])
            
            # 条件2：目标出现时间在丢失时间之后（新出现的或重新出现的）
            if track_id in self.tracked_persons:
                is_new_or_reappeared = (
                    self.tracked_persons[track_id]['first_seen_time'] > self.target_lost_time or
                    track_id == self.current_tracking_id  # 原始目标重新出现
                )
            else:
                is_new_or_reappeared = True
            
            # 条件3：避免匹配已经存在一段时间的其他目标
            is_not_old_target = (
                track_id == self.current_tracking_id or  # 原始目标总是可以匹配
                (track_id in self.tracked_persons and 
                 timestamp - self.tracked_persons[track_id]['first_seen_time'] < 3.0)  # 3秒内出现的新目标
            )
            
            if not is_currently_tracked and is_new_or_reappeared and is_not_old_target:
                candidate_tracks.append(track)
        
        if not candidate_tracks:
            # 检查原始目标是否重新出现
            original_target_present = any(track['track_id'] == self.current_tracking_id for track in current_tracks)
            if original_target_present:
                self.get_logger().info(f"目标 {self.current_tracking_id} 重新出现，直接恢复跟踪")
                self.target_lost_time = None
                return self.current_tracking_id
            
            self.get_logger().info(f"目标 {self.current_tracking_id} 找回: 当前帧无合适候选目标")
            return None
        
        # 特征匹配
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
                
                # 如果是原始目标重新出现，优先选择
                if track_id == self.current_tracking_id:
                    best_match_id = track_id
                    best_similarity = max(similarity, 0.9)
                    self.get_logger().info(f"优先选择原始目标 ID {track_id}")
                    break
                
                if similarity > self.reid_similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = track_id
        
        if best_match_id is not None:
            self.get_logger().info(
                f"目标 {self.current_tracking_id} 找回成功! 匹配ID: {best_match_id}, 相似度: {best_similarity:.3f}"
            )
            target_bbox = next(t['bbox'] for t in candidate_tracks if t['track_id'] == best_match_id)
            self.save_tracked_target(self.current_tracking_id, target_bbox, image, timestamp)
            target.is_recovered = True
            target.lost_frames = 0
            
            # 如果是新匹配的目标，标记为切换状态
            if best_match_id != self.current_tracking_id:
                if best_match_id in self.tracked_targets:
                    self.tracked_targets[best_match_id].switch_to_new_id(best_match_id)
                    self.tracked_targets[best_match_id].original_track_id = self.current_tracking_id
            
            self.target_lost_time = None
            return best_match_id
        
        return None

    def estimate_depth_from_bbox_height(self, bbox_height_pixels: float) -> float:
        """基于边界框高度估计深度"""
        return (self.real_person_height * self.fy) / bbox_height_pixels

    def _get_keypoints_depth(self, keypoints: np.ndarray) -> float:
        """从关键点获取深度"""
        valid_kps_indices = [
            self.KEYPOINT_NAMES['LEFT_SHOULDER'],
            self.KEYPOINT_NAMES['RIGHT_SHOULDER'],
            self.KEYPOINT_NAMES['LEFT_HIP'],
            self.KEYPOINT_NAMES['RIGHT_HIP']
        ]
        
        valid_depths = []
        with self.depth_lock:
            if self.depth_image is None:
                return 0.0
                
            for idx in valid_kps_indices:
                if idx >= len(keypoints) or np.isnan(keypoints[idx]).any() or (keypoints[idx][0] == 0 and keypoints[idx][1] == 0):
                    continue
                x, y = keypoints[idx].astype(int)
                if 0 <= x < self.depth_image.shape[1] and 0 <= y < self.depth_image.shape[0]:
                    depth = self.depth_image[y, x] / 1000.0
                    if 0.5 < depth < 8.0:
                        valid_depths.append(depth)
        
        return np.median(valid_depths) if valid_depths else 0.0

    def compute_body_depth(self, bbox: List[float], keypoints: np.ndarray, track_id: int) -> float:
        """融合关键点深度和边界框高度估计"""
        keypoints_depth = self._get_keypoints_depth(keypoints)
        
        _, y1, _, y2 = bbox
        bbox_height = y2 - y1
        bbox_estimated_depth = self.estimate_depth_from_bbox_height(bbox_height)
        
        if keypoints_depth <= 0:
            final_depth = bbox_estimated_depth
        else:
            final_depth = (keypoints_depth * 0.7 + bbox_estimated_depth * 0.3)
        
        if track_id not in self.depth_filters:
            self.depth_filters[track_id] = DepthFilter()
        
        self.depth_filters[track_id].add_depth(final_depth)
        return self.depth_filters[track_id].get_filtered_depth()

    def depth_callback(self, msg):
        with self.depth_lock:
            try:
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            except Exception as e:
                self.get_logger().error(f"Depth image conversion error: {str(e)}")

    def image_callback(self, msg):
        current_time = time.time()
        if current_time - self.last_process_time < self.min_process_interval:
            return
        
        total_start_time = time.time()
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
                    self.depth_filters[track_id] = DepthFilter()
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

            # 处理丢失目标
            if self.current_tracking_id is not None and self.current_tracking_id not in current_track_ids:
                if self.current_tracking_id in self.tracked_targets:
                    self.tracked_targets[self.current_tracking_id].mark_lost()
                    if self.target_lost_time is None:
                        self.target_lost_time = current_time
                    self.get_logger().info(f"目标 {self.current_tracking_id} 丢失，已丢失 {self.tracked_targets[self.current_tracking_id].lost_frames} 帧")
            
            # 尝试找回丢失的目标
            recovered_id = self.try_recover_lost_target(tracks, cv_image, current_time)
            if recovered_id is not None:
                self.current_tracking_id = recovered_id
                if recovered_id in self.tracked_persons:
                    self.tracked_persons[recovered_id]['is_tracking'] = True
                    self.tracked_persons[recovered_id]['tracking_start_time'] = current_time

            # 清理长时间未出现的跟踪目标
            self.cleanup_old_tracks(current_time, current_track_ids)

            # 可视化并发布结果
            self.visualize_results(cv_image, tracks)
            self.publish_person_positions(tracks, msg.header)
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
                    if track_id in self.depth_filters:
                        del self.depth_filters[track_id]
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
                    self.get_logger().info(f"清理长时间未出现的跟踪目标 ID: {track_id}")

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
        """可视化跟踪结果"""
        display_image = image.copy()
        
        for track in tracks:
            track_id = track['track_id']
            x1, y1, x2, y2 = track['bbox']

            # 确保只有一个目标被跟踪
            is_tracking = (track_id == self.current_tracking_id and 
                          track_id in self.tracked_persons and 
                          self.tracked_persons[track_id]['is_tracking'])
            
            if is_tracking:
                tracking_time = time.time() - self.tracked_persons[track_id]['tracking_start_time']
                in_protection_period = tracking_time < self.tracking_protection_time
            else:
                in_protection_period = False

            # 设置颜色
            if is_tracking:
                if in_protection_period:
                    box_color = (255, 165, 0)  # 橙色 - 保护期内
                    text_color = (255, 165, 0)
                else:
                    box_color = (255, 0, 0)    # 红色 - 跟踪中
                    text_color = (255, 0, 0)
            else:
                box_color = (0, 255, 0)        # 绿色 - 未跟踪
                text_color = (0, 255, 0)

            # 绘制边界框
            cv2.rectangle(display_image, (x1, y1), (x2, y2), box_color, 2)

            # 显示跟踪状态信息
            if track_id in self.tracked_persons:
                person = self.tracked_persons[track_id]
                if is_tracking:
                    if in_protection_period:
                        protection_left = self.tracking_protection_time - tracking_time
                        label = f"ID: {track_id} Protection: {protection_left:.1f}s"
                    else:
                        label = f"ID: {track_id} Tracking: {tracking_time:.1f}s"
                else:
                    confirm_count = sum(self.hands_up_history.get(track_id, []))
                    label = f"ID: {track_id} Ready ({confirm_count}/{self.hands_up_confirm_frames})"
            else:
                label = f"ID: {track_id} Ready"

            # 绘制文本
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display_image, (x1, y1 - text_height - 5), (x1 + text_width, y1), (0, 0, 0), -1)
            cv2.putText(display_image, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

            # 绘制关键点和骨架
            keypoints = track['keypoints']
            for kp in keypoints:
                if not np.isnan(kp).any() and not (kp[0] == 0 and kp[1] == 0):
                    x, y = kp.astype(int)
                    cv2.circle(display_image, (x, y), 3, (0, 0, 255), -1)

            for connection in self.skeleton:
                idx1, idx2 = connection
                idx1 -= 1
                idx2 -= 1
                if (idx1 < len(keypoints) and idx2 < len(keypoints) and
                    not np.isnan(keypoints[idx1]).any() and not np.isnan(keypoints[idx2]).any() and
                    not (keypoints[idx1][0] == 0 and keypoints[idx1][1] == 0) and
                    not (keypoints[idx2][0] == 0 and keypoints[idx2][1] == 0)):
                    pt1 = keypoints[idx1].astype(int)
                    pt2 = keypoints[idx2].astype(int)
                    cv2.line(display_image, pt1, pt2, (0, 255, 255), 2)

        image[:] = display_image[:]
        
    def publish_tracked_keypoints(self, tracks: List[Dict], header):
        """发布边界框的四个角点，包含目标状态信息"""
        # 检查当前是否有跟踪目标
        has_tracking_target = False
        
        for track in tracks:
            track_id = track['track_id']
            if track_id in self.tracked_persons and self.tracked_persons[track_id]['is_tracking']:
                has_tracking_target = True
                x1, y1, x2, y2 = track['bbox']
                
                polygon_msg = PolygonStamped()
                polygon_msg.header = header
                polygon_msg.header.frame_id = "camera_link"
                
                # 添加目标状态信息到消息中
                # 使用第一个点存储状态信息：x=track_id, y=1(正常), z=0(保留)
                points = [
                    Point32(x=float(track_id), y=1.0, z=0.0),  # 状态点：y=1表示目标正常
                    Point32(x=float(x1), y=float(y1), z=0.0),   # 左上角
                    Point32(x=float(x2), y=float(y2), z=0.0),   # 右下角
                ]
                
                polygon_msg.polygon.points = points
                self.keypoint_tracks_pub.publish(polygon_msg)
        
        # 如果当前应该有跟踪目标但目标丢失了
        if self.current_tracking_id is not None and not has_tracking_target:
            polygon_msg = PolygonStamped()
            polygon_msg.header = header
            polygon_msg.header.frame_id = "camera_link"
            
            # 发布目标丢失状态：y=0表示目标丢失
            points = [
                Point32(x=float(self.current_tracking_id), y=0.0, z=0.0),  # 状态点：y=0表示目标丢失
                Point32(x=0.0, y=0.0, z=0.0),  # 无效坐标
                Point32(x=0.0, y=0.0, z=0.0),  # 无效坐标
            ]
            
            polygon_msg.polygon.points = points
            self.keypoint_tracks_pub.publish(polygon_msg)
            self.get_logger().info(f"发布目标丢失状态: ID {self.current_tracking_id}")



    def publish_person_positions(self, tracks: List[Dict], header):
        for track in tracks:
            track_id = track['track_id']
            if track_id in self.tracked_persons and not self.tracked_persons[track_id]['is_tracking']:
                continue
                
            x1, y1, x2, y2 = track['bbox']
            keypoints = track['keypoints']
            
            depth = self.compute_body_depth([x1, y1, x2, y2], keypoints, track_id)
            
            if depth <= 0:
                continue
                
            # 计算中心点坐标
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # 转换为3D坐标
            x = (center_x - self.cx) * depth / self.fx
            y = (center_y - self.cy) * depth / self.fy
            z = depth
            
            # 创建并发布PointStamped消息
            point_msg = PointStamped()
            point_msg.header = header
            point_msg.header.frame_id = "camera_link"
            point_msg.point.x = x
            point_msg.point.y = y
            point_msg.point.z = z
            
            self.person_point_pub.publish(point_msg)

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