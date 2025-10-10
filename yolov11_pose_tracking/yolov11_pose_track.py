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
from typing import List, Dict, Optional, Tuple
from collections import deque
import threading 
import copy

# 导入RDK YOLO模型和BOTSORT跟踪器
from .YOLO_Pose import YOLO11_Pose
from .BOTSort_rdk import BOTSORT, OSNetReID

class DepthFilter:
    """深度滤波器 - 优化内存分配"""
    __slots__ = ('depth_window_size', 'depth_threshold', 'depth_window')
    
    def __init__(self, depth_window_size=5, depth_threshold=0.5):
        self.depth_window_size = depth_window_size
        self.depth_threshold = depth_threshold
        self.depth_window = deque(maxlen=depth_window_size)

    def add_depth(self, depth):
        self.depth_window.append(depth)

    def get_filtered_depth(self):
        if not self.depth_window:
            return 0.0
        return np.median(self.depth_window) if len(self.depth_window) > 1 else self.depth_window[-1]

class TrackedTarget:
    """跟踪目标信息类 - 使用__slots__优化内存"""
    __slots__ = ('track_id', 'bbox', 'feature', 'height_pixels', 'first_seen_time', 
                 'last_seen_time', 'last_update_time', 'lost_frames', 'is_recovered', 
                 'is_switched', 'original_track_id', 'recovery_time', 'update_paused')
    
    def __init__(self, track_id: int, bbox: List[float], feature: np.ndarray, 
                 height_pixels: float, timestamp: float):
        self.track_id = track_id
        self.bbox = bbox
        self.feature = feature
        self.height_pixels = height_pixels
        self.first_seen_time = timestamp
        self.last_seen_time = timestamp
        self.last_update_time = timestamp
        self.lost_frames = 0
        self.is_recovered = False
        self.is_switched = False
        self.original_track_id = track_id
        self.recovery_time = None
        self.update_paused = False
    
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
        self.update_paused = True
        self.recovery_time = None
    
    def mark_recovered(self, timestamp: float):
        """标记目标找回"""
        self.is_recovered = True
        self.lost_frames = 0
        self.recovery_time = timestamp
    
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

        # 声明参数
        self._declare_parameters()
        
        # 获取参数
        self._get_parameters()
        
        self.min_process_interval = 1.0 / self.max_processing_fps
        self.last_process_time = time.time()

        # 初始化模型和组件
        self._initialize_components()
        
        # 初始化变量
        self._initialize_variables()
        
        self.get_logger().info("YOLOv11 Pose Node initialized with ReID recovery (Optimized)")
        self.print_parameters()

    def _declare_parameters(self):
        """声明所有参数"""
        self.declare_parameter('model_path', '')
        self.declare_parameter('conf_threshold', 0.3)
        self.declare_parameter('kpt_conf_threshold', 0.25)
        self.declare_parameter('real_person_height', 1.7)
        self.declare_parameter('max_processing_fps', 15)
        self.declare_parameter('hands_up_confirm_frames', 3)
        self.declare_parameter('tracking_protection_time', 5.0)
        self.declare_parameter('reid_similarity_threshold', 0.8)
        self.declare_parameter('feature_update_interval', 1.0)
        self.declare_parameter('reid_model_path', 'osnet_64x128_nv12.bin')

    def _get_parameters(self):
        """获取参数值"""
        self.conf_threshold = self.get_parameter('conf_threshold').value
        self.kpt_conf_threshold = self.get_parameter('kpt_conf_threshold').value
        self.real_person_height = self.get_parameter('real_person_height').value
        self.max_processing_fps = self.get_parameter('max_processing_fps').value
        self.hands_up_confirm_frames = self.get_parameter('hands_up_confirm_frames').value
        self.tracking_protection_time = self.get_parameter('tracking_protection_time').value
        self.reid_similarity_threshold = self.get_parameter('reid_similarity_threshold').value
        self.feature_update_interval = self.get_parameter('feature_update_interval').value
        # reid_model_path = self.get_parameter('reid_model_path').value

    def _initialize_components(self):
        """初始化模型和跟踪器"""
        model_path = self.get_parameter('model_path').value
        reid_model_path = self.get_parameter('reid_model_path').value
        
        # 加载YOLOv11 pose模型
        self.model = YOLO11_Pose(model_path, self.conf_threshold, 0.45)
        
        # 初始化ReID编码器
        self.reid_encoder = None
        try:
            self.reid_encoder = OSNetReID(reid_model_path)
            self.get_logger().info(f"ReID模型加载成功: {reid_model_path}")
        except Exception as e:
            self.get_logger().error(f"ReID模型加载失败: {e}")
        
        # 初始化BOTSORT跟踪器
        tracker_args = {
            'track_high_thresh': 0.25,
            'track_low_thresh': 0.1,
            'new_track_thresh': 0.25,
            'track_buffer': 10,
            'match_thresh': 0.68,
            'fuse_score': False,
            'gmc_method': 'sparseOptFlow',
            'proximity_thresh': 0.5,
            'appearance_thresh': 0.7,
            'with_reid': False,
            'reid_model_path': reid_model_path
        }
        self.tracker = BOTSORT(tracker_args)
        
        # 初始化骨架连接
        self.skeleton = [
            (16, 14), (14, 12), (17, 15), (15, 13), (12, 13),
            (6, 12), (7, 13), (6, 7), (6, 8), (7, 9),
            (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
            (2, 4), (3, 5), (4, 6), (5, 7)
        ]
        
        # 初始化CV bridge
        self.bridge = CvBridge()
        
        # 创建订阅和发布
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, 10)
        self.detect_pose_pub = self.create_publisher(Image, 'tracks', 10)
        self.person_point_pub = self.create_publisher(PointStamped, '/person_positions', 10)
        self.keypoint_tracks_pub = self.create_publisher(PolygonStamped, '/keypoint_tracks', 10)

    def _initialize_variables(self):
        """初始化变量"""
        # 跟踪和深度相关变量
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
        
        # 目标丢失时间记录
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

    def print_parameters(self):
        """打印参数信息"""
        self.get_logger().info("===== 参数配置信息 =====")
        self.get_logger().info(f"置信度阈值: {self.conf_threshold}")
        self.get_logger().info(f"关键点置信度阈值: {self.kpt_conf_threshold}")
        self.get_logger().info(f"真实人身高: {self.real_person_height}m")
        self.get_logger().info(f"最大处理帧率: {self.max_processing_fps}FPS")
        self.get_logger().info(f"举手确认帧数: {self.hands_up_confirm_frames}")
        self.get_logger().info(f"跟踪保护时间: {self.tracking_protection_time}s")
        self.get_logger().info(f"ReID相似度阈值: {self.reid_similarity_threshold}")
        self.get_logger().info(f"特征更新间隔: {self.feature_update_interval}s")
        self.get_logger().info("=========================")

    def extract_feature_from_bbox(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """从边界框提取特征 - 优化内存分配"""
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        
        # 边界检查
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
                return self.reid_encoder.extract_feature(crop)
            else:
                return np.zeros(512, dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f"Feature extraction failed: {e}")
            return np.zeros(512, dtype=np.float32)
        
    def save_tracked_target(self, track_id: int, bbox: List[float], image: np.ndarray, timestamp: float):
        """保存跟踪目标信息 - 优化内存分配"""
        if track_id not in self.tracked_targets:
            feature = self.extract_feature_from_bbox(image, bbox)
            height_pixels = bbox[3] - bbox[1]
            self.tracked_targets[track_id] = TrackedTarget(track_id, bbox, feature, height_pixels, timestamp)
            return
        
        target = self.tracked_targets[track_id]
        
        # 检查是否处于找回后的冷却期
        if target.recovery_time is not None:
            time_since_recovery = timestamp - target.recovery_time
            if time_since_recovery < 1.0:
                target.bbox = bbox
                target.height_pixels = bbox[3] - bbox[1]
                target.last_seen_time = timestamp
                target.lost_frames = 0
                target.update_paused = True
                return
            else:
                target.update_paused = False
                target.recovery_time = None
        
        # 如果更新被暂停，跳过特征更新
        if target.update_paused:
            target.bbox = bbox
            target.height_pixels = bbox[3] - bbox[1]
            target.last_seen_time = timestamp
            target.lost_frames = 0
            return
        
        # 正常更新频率控制
        time_since_update = timestamp - target.last_update_time
        if time_since_update < self.feature_update_interval:
            target.bbox = bbox
            target.height_pixels = bbox[3] - bbox[1]
            target.last_seen_time = timestamp
            target.lost_frames = 0
            target.is_recovered = False
            return
        
        # 完整更新（包括特征）
        feature = self.extract_feature_from_bbox(image, bbox)
        height_pixels = bbox[3] - bbox[1]
        target.update(bbox, feature, height_pixels, timestamp)

    def try_recover_lost_target(self, current_tracks: List[Dict], image: np.ndarray, timestamp: float) -> Optional[int]:
        """立即尝试找回丢失的跟踪目标"""
        if self.current_tracking_id is None or self.current_tracking_id not in self.tracked_targets:
            return None
        
        target = self.tracked_targets[self.current_tracking_id]
        
        # 立即记录丢失时间
        if self.target_lost_time is None:
            self.target_lost_time = timestamp
            self.get_logger().info(f"目标 {self.current_tracking_id} 丢失，开始立即ReID匹配找回")
        
        # 准备候选目标
        candidate_tracks = []
        for track in current_tracks:
            track_id = track['track_id']
            is_currently_tracked = (
                track_id in self.tracked_persons and 
                self.tracked_persons[track_id]['is_tracking'] and
                track_id != self.current_tracking_id
            )
            
            if not is_currently_tracked:
                candidate_tracks.append(track)
        
        if not candidate_tracks:
            return None
        
        # ReID匹配
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
                
                if similarity >= self.reid_similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = track_id
        
        if best_match_id is not None:
            self.get_logger().info(
                f"目标 {self.current_tracking_id} ReID找回成功! 匹配ID: {best_match_id}, 相似度: {best_similarity:.3f}"
            )
            
            target_bbox = next(t['bbox'] for t in candidate_tracks if t['track_id'] == best_match_id)
            target.mark_recovered(timestamp)
            self.save_tracked_target(self.current_tracking_id, target_bbox, image, timestamp)
            
            recovered_id = best_match_id
            
            if best_match_id != self.current_tracking_id:
                if best_match_id in self.tracked_targets:
                    self.tracked_targets[best_match_id].switch_to_new_id(best_match_id)
                    self.tracked_targets[best_match_id].original_track_id = self.current_tracking_id
                    self.tracked_targets[best_match_id].mark_recovered(timestamp)
            
            self.target_lost_time = None
            
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
            
            if similarity >= 0.8:
                self.get_logger().info(f"ReID验证成功: ID {track_id}, 相似度: {similarity:.3f}")
                
                target.mark_recovered(timestamp)
                self.save_tracked_target(target.track_id, bbox, image, timestamp)
                self.target_lost_time = None
                
                if track_id in self.tracked_persons:
                    self.tracked_persons[track_id]['is_tracking'] = True
                    self.tracked_persons[track_id]['last_seen_time'] = timestamp
                    if track_id == target.track_id:
                        pass
                    else:
                        self.tracked_persons[track_id]['tracking_start_time'] = timestamp
                
                return track_id
            else:
                self.get_logger().warning(f"ReID验证失败: ID {track_id}, 相似度: {similarity:.3f}")
        
        return None

    def estimate_depth_from_bbox_height(self, bbox_height_pixels: float) -> float:
        """基于边界框高度估计深度"""
        return (self.real_person_height * self.fy) / bbox_height_pixels

    def _get_keypoints_depth(self, keypoints: np.ndarray) -> float:
        """从关键点获取深度 - 优化计算"""
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
        """深度图像回调 - 优化内存分配"""
        with self.depth_lock:
            try:
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            except Exception as e:
                self.get_logger().error(f"Depth image conversion error: {str(e)}")

    def image_callback(self, msg):
        """图像回调 - 优化性能"""
        current_time = time.time()
        if current_time - self.last_process_time < self.min_process_interval:
            return
        
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
            self._update_tracking_state(tracks, cv_image, current_time)
            
            # 清理长时间未出现的跟踪目标
            self._cleanup_old_tracks(current_time, set(track['track_id'] for track in tracks))

            # 可视化并发布结果
            self._publish_results(cv_image, tracks, msg.header)

        except Exception as e:
            self.get_logger().error(f"Image processing error: {str(e)}")

    def _update_tracking_state(self, tracks: List[Dict], cv_image: np.ndarray, current_time: float):
        """更新跟踪状态 - 提取为独立方法"""
        current_track_ids = set()
        
        for track in tracks:
            track_id = track['track_id']
            current_track_ids.add(track_id)
            
            if track_id not in self.tracked_persons:
                self._initialize_new_track(track_id, current_time)
            else:
                self.tracked_persons[track_id]['last_seen_time'] = current_time

            person = self.tracked_persons[track_id]
            hands_up = self.is_hands_up(track)
            
            self.hands_up_history[track_id].append(hands_up)
            hands_up_confirmed = sum(self.hands_up_history[track_id]) >= self.hands_up_confirm_frames
            
            if self.current_tracking_id is None:
                self._handle_new_tracking(track_id, person, hands_up_confirmed, track, cv_image, current_time)
            elif self.current_tracking_id == track_id:
                self._handle_current_tracking(track_id, person, hands_up_confirmed, track, cv_image, current_time)

        # 处理丢失目标
        self._handle_lost_targets(current_track_ids, tracks, cv_image, current_time)

    def _initialize_new_track(self, track_id: int, current_time: float):
        """初始化新跟踪目标"""
        self.tracked_persons[track_id] = {
            'is_tracking': False,
            'tracking_start_time': 0.0,
            'last_hands_up_time': 0.0,
            'first_seen_time': current_time,
            'last_seen_time': current_time
        }
        self.depth_filters[track_id] = DepthFilter()
        self.hands_up_history[track_id] = deque(maxlen=self.hands_up_confirm_frames)

    def _handle_new_tracking(self, track_id: int, person: Dict, hands_up_confirmed: bool, 
                           track: Dict, cv_image: np.ndarray, current_time: float):
        """处理新跟踪目标"""
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

    def _handle_current_tracking(self, track_id: int, person: Dict, hands_up_confirmed: bool, 
                               track: Dict, cv_image: np.ndarray, current_time: float):
        """处理当前跟踪目标"""
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

    def _handle_lost_targets(self, current_track_ids: set, tracks: List[Dict], 
                           cv_image: np.ndarray, current_time: float):
        """处理丢失目标"""
        if self.current_tracking_id is not None and self.current_tracking_id not in current_track_ids:
            if self.current_tracking_id in self.tracked_targets:
                self.tracked_targets[self.current_tracking_id].mark_lost()
                
                if self.current_tracking_id in self.tracked_persons:
                    self.tracked_persons[self.current_tracking_id]['is_tracking'] = False
                
                if self.target_lost_time is None:
                    self.target_lost_time = current_time
                    self.get_logger().warning(f"目标 {self.current_tracking_id} 丢失，立即启动ReID找回")
                
                recovered_id = self.try_recover_lost_target(tracks, cv_image, current_time)
                if recovered_id is not None:
                    self.current_tracking_id = recovered_id
                    if recovered_id in self.tracked_persons:
                        self.tracked_persons[recovered_id]['is_tracking'] = True
                        self.tracked_persons[recovered_id]['tracking_start_time'] = current_time
                        self.get_logger().info(f"ReID找回成功，切换到新ID: {recovered_id}")
                else:
                    self.get_logger().warning(f"目标 {self.current_tracking_id} ReID找回失败，保持丢失状态")
        
        elif self.current_tracking_id is not None and self.current_tracking_id in current_track_ids:
            if self.target_lost_time is not None:
                self.get_logger().info(f"目标 {self.current_tracking_id} 重新出现，进行ReID验证")
                track = next(t for t in tracks if t['track_id'] == self.current_tracking_id)
                verified_id = self._verify_target_with_reid(
                    self.tracked_targets[self.current_tracking_id], track, cv_image, current_time
                )
                
                if verified_id is not None:
                    self.target_lost_time = None
                    self.get_logger().info(f"目标 {self.current_tracking_id} ReID验证成功，继续跟踪")
                    
                    if verified_id in self.tracked_persons:
                        self.tracked_persons[verified_id]['is_tracking'] = True
                        self.tracked_persons[verified_id]['last_seen_time'] = current_time

    def _cleanup_old_tracks(self, current_time: float, current_track_ids: set):
        """清理长时间未出现的跟踪目标"""
        max_track_age = 5.0
        
        for track_id in list(self.tracked_persons.keys()):
            if track_id == self.current_tracking_id:
                continue
                
            if track_id not in current_track_ids:
                last_seen = self.tracked_persons[track_id]['last_seen_time']
                if current_time - last_seen > max_track_age:
                    self._remove_track(track_id)

    def _remove_track(self, track_id: int):
        """移除跟踪目标"""
        if track_id in self.tracked_persons:
            del self.tracked_persons[track_id]
        if track_id in self.depth_filters:
            del self.depth_filters[track_id]
        if track_id in self.hands_up_history:
            del self.hands_up_history[track_id]
        if track_id in self.tracked_targets:
            target = self.tracked_targets[track_id]
            if target.is_switched and target.original_track_id not in self.tracked_targets:
                original_id = target.original_track_id
                self.tracked_targets[original_id] = copy.deepcopy(target)
                self.tracked_targets[original_id].track_id = original_id
                self.tracked_targets[original_id].is_switched = False
            
            del self.tracked_targets[track_id]

    def is_hands_up(self, track: Dict) -> bool:
        """举手检测 - 优化计算"""
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
            left_hand_up = (left_wrist[1] < left_shoulder[1] and
                           left_wrist[1] < nose[1] and
                           abs(left_wrist[0] - left_shoulder[0]) < 80)

        right_hand_up = False
        if (has_valid_keypoint(self.KEYPOINT_NAMES['RIGHT_WRIST']) and 
            has_valid_keypoint(self.KEYPOINT_NAMES['RIGHT_ELBOW'])):
            right_wrist = keypoints[self.KEYPOINT_NAMES['RIGHT_WRIST']]
            right_hand_up = (right_wrist[1] < right_shoulder[1] and
                            right_wrist[1] < nose[1] and
                            abs(right_wrist[0] - right_shoulder[0]) < 80)

        return left_hand_up or right_hand_up
        
    def _publish_results(self, image: np.ndarray, tracks: List[Dict], header):
        """发布结果 - 合并可视化"""
        self.visualize_results(image, tracks)
        self.publish_person_positions(tracks, header)
        self.publish_tracked_keypoints(tracks, header)

        # 只在有订阅者时才进行可视化发布
        if self.detect_pose_pub.get_subscription_count() > 0:
            detect_pose_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            detect_pose_msg.header = header
            self.detect_pose_pub.publish(detect_pose_msg)

    def visualize_results(self, image: np.ndarray, tracks: List[Dict]):
        """简化版可视化跟踪结果 - 优化绘制性能"""
        display_image = image.copy()
        
        for track in tracks:
            track_id = track['track_id']
            x1, y1, x2, y2 = track['bbox']
            confidence = track['conf']

            # 确定跟踪状态和颜色
            is_tracking = (track_id == self.current_tracking_id and 
                        track_id in self.tracked_persons and 
                        self.tracked_persons[track_id]['is_tracking'])
            
            color = (255, 0, 0) if is_tracking else (0, 255, 0)
            thickness = 3 if is_tracking else 2

            # 绘制边界框
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, thickness)
            
            # 绘制标签
            label = f"ID:{track_id} {confidence:.2f}"
            if is_tracking:
                label = f"TRACKING {label}"
            
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(display_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(display_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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

        # 更新图像
        image[:] = display_image

    def publish_person_positions(self, tracks: List[Dict], header):
        """发布人员位置信息 - 优化计算"""
        for track in tracks:
            track_id = track['track_id']
            
            if (track_id == self.current_tracking_id and 
                track_id in self.tracked_persons and 
                self.tracked_persons[track_id]['is_tracking']):
                
                depth = self.compute_body_depth(track['bbox'], track['keypoints'], track_id)
                
                if depth > 0:
                    x1, y1, x2, y2 = track['bbox']
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    point_msg = PointStamped()
                    point_msg.header = header
                    point_msg.point.x = (center_x - self.cx) * depth / self.fx
                    point_msg.point.y = (center_y - self.cy) * depth / self.fy
                    point_msg.point.z = depth
                    
                    self.person_point_pub.publish(point_msg)
                    break

    def publish_tracked_keypoints(self, tracks: List[Dict], header):
        """发布边界框和肩部关键点坐标和置信度 - 简化优化版"""
        current_tracking_id = self.current_tracking_id
        
        # 查找当前跟踪的目标
        tracking_target = None
        for track in tracks:
            track_id = track['track_id']
            if (current_tracking_id is not None and track_id == current_tracking_id) or \
            (track_id in self.tracked_persons and self.tracked_persons[track_id]['is_tracking']):
                tracking_target = track
                break
        
        polygon_msg = PolygonStamped()
        polygon_msg.header = header
        polygon_msg.header.frame_id = "camera_link"
        
        if tracking_target:
            # 发布正常跟踪状态
            track_id = tracking_target['track_id']
            x1, y1, x2, y2 = tracking_target['bbox']
            keypoints = tracking_target['keypoints']
            keypoints_conf = tracking_target['keypoints_conf']
            
            # 构建消息点：状态信息 + 边界框 + 肩部关键点
            points = [
                Point32(x=float(track_id), y=1.0, z=2.0),  # 状态点
                Point32(x=float(x1), y=float(y1), z=0.0),   # 边界框左上
                Point32(x=float(x2), y=float(y2), z=0.0),   # 边界框右下
            ]
            
            # 添加肩部关键点
            shoulder_indices = [
                self.KEYPOINT_NAMES['LEFT_SHOULDER'],
                self.KEYPOINT_NAMES['RIGHT_SHOULDER']
            ]
            
            for kp_index in shoulder_indices:
                if kp_index < len(keypoints):
                    kp = keypoints[kp_index]
                    conf = keypoints_conf[kp_index] if kp_index < len(keypoints_conf) else 0.0
                    
                    if not np.isnan(kp).any() and not (kp[0] == 0 and kp[1] == 0):
                        points.append(Point32(x=float(kp[0]), y=float(kp[1]), z=float(conf)))
                    else:
                        points.append(Point32(x=0.0, y=0.0, z=0.0))
                else:
                    points.append(Point32(x=0.0, y=0.0, z=0.0))
            
            polygon_msg.polygon.points = points
            
        elif current_tracking_id is not None:
            # 发布目标丢失状态
            points = [
                Point32(x=float(current_tracking_id), y=0.0, z=0.0),  # 状态点：y=0表示丢失
                Point32(x=0.0, y=0.0, z=0.0),
                Point32(x=0.0, y=0.0, z=0.0),
                Point32(x=0.0, y=0.0, z=0.0),
                Point32(x=0.0, y=0.0, z=0.0)
            ]
            polygon_msg.polygon.points = points
            self.get_logger().info(f"发布目标丢失状态: ID {current_tracking_id}")
        
        else:
            # 无跟踪目标状态
            points = [Point32(x=0.0, y=0.0, z=0.0) for _ in range(5)]  # 5个零值点
            polygon_msg.polygon.points = points
        
        # 发布消息
        self.keypoint_tracks_pub.publish(polygon_msg)

def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = Yolov11PoseNode()
        executor = MultiThreadedExecutor()
        executor.add_node(node)
        
        try:
            executor.spin()
        finally:
            executor.shutdown()
            node.destroy_node()
            
    except Exception as e:
        print(f"Node initialization failed: {e}")
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()