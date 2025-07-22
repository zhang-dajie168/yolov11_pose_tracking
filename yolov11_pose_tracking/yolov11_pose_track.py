#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import os
import time
from typing import List, Dict, Tuple, Deque
from collections import deque
from ultralytics.trackers.byte_tracker  import BYTETracker,STrack
from argparse import Namespace
import threading
import yaml
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.executors import MultiThreadedExecutor

class Yolov11PoseNode(Node):
    def __init__(self):
        super().__init__('yolov11_pose_node')

        # 打印所有可用参数
        self.get_logger().info("Available parameters:")
        for param_name in self._parameters:
            self.get_logger().info(f"  - {param_name}")

        # Camera intrinsics (Orbbec Gemini 335L 640x480)
        self.fx = 367.21
        self.fy = 316.44
        self.cx = 367.20
        self.cy = 244.60

        # Declare and get parameters
        self.declare_parameter('model_path', '')
        self.declare_parameter('conf_threshold', 0.25)
        self.declare_parameter('kpt_conf_threshold', 0.25)
        self.declare_parameter('tracking_threshold', 0.5)
        self.declare_parameter('track_buffer', 30)
        self.declare_parameter('match_threshold', 0.9)

        # 获取参数值
        model_path = self.get_parameter('model_path').value
        conf_threshold = self.get_parameter('conf_threshold').value
        kpt_conf_threshold = self.get_parameter('kpt_conf_threshold').value

        # 如果参数为空，尝试从环境变量获取
        if not model_path:
            self.get_logger().warn("model_path parameter is empty!")
            model_path = os.getenv('YOLO_MODEL_PATH', '')

        # 检查参数值
        self.get_logger().info(f"Loaded parameters:")
        self.get_logger().info(f"  model_path: {model_path}")
        self.get_logger().info(f"  conf_threshold: {conf_threshold}")
        self.get_logger().info(f"  kpt_conf_threshold: {kpt_conf_threshold}")
        self.get_logger().info(f"  tracking_threshold: {self.get_parameter('tracking_threshold').value}")
        self.get_logger().info(f"  track_buffer: {self.get_parameter('track_buffer').value}")
        self.get_logger().info(f"  match_threshold: {self.get_parameter('match_threshold').value}")

        # 验证模型文件是否存在
        if not model_path or not os.path.isfile(model_path):
            # 尝试默认路径
            default_path = "/home/peng/Ebike_Human_Follower/src/yolov11_pose_tracking/models/yolo11n-pose.pt"
            if os.path.isfile(default_path):
                self.get_logger().warn(f"Using default model path: {default_path}")
                model_path = default_path
            else:
                self.get_logger().error(f"Model file not found: {model_path}")
                raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load YOLOv11 pose model
        self.get_logger().info(f"Loading model from {model_path}")
        self.model = YOLO(model_path, task='pose')  # 显式指定任务类型
        self.get_logger().info("Model loaded successfully")
        
        # Initialize BYTETracker
        # 修改 BYTETracker 初始化部分
        tracker_params = Namespace(
            track_thresh=self.get_parameter('tracking_threshold').value,
            track_buffer=self.get_parameter('track_buffer').value,
            match_thresh=self.get_parameter('match_threshold').value,
            mot20=False,  # 添加必要的默认参数
            track_high_thresh=0.6,
            track_low_thresh=0.1,
            new_track_thresh=0.7,
            frame_rate=30.0  # 添加帧率参数
        )
        self.tracker = BYTETracker(tracker_params)
        
        # Initialize skeleton connections
        self.skeleton = [
            (16, 14), (14, 12), (17, 15), (15, 13), (12, 13),
            (6, 12), (7, 13), (6, 7), (6, 8), (7, 9),
            (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
            (2, 4), (3, 5), (4, 6), (5, 7)
        ]
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Create subscriptions
        self.image_sub = self.create_subscription(
            Image, 
            '/camera/color/image_raw', 
            self.image_callback, 
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image, 
            '/camera/depth/image_raw', 
            self.depth_callback, 
            10
        )
        
        # Create publishers
        self.detect_pose_pub = self.create_publisher(Image, 'detect_pose', 10)
        self.person_point_pub = self.create_publisher(PointStamped, '/person_positions', 10)
        self.keypoints_cloud_pub = self.create_publisher(PointCloud2, '/keypoints_cloud', 10)
        
        # Tracked persons dictionary
        self.tracked_persons: Dict[int, Dict] = {}
        
        # Depth image storage
        self.depth_image = None
        self.depth_lock = threading.Lock()
        
        # Confidence thresholds
        self.conf_threshold = conf_threshold
        self.kpt_conf_threshold = kpt_conf_threshold
        
        self.get_logger().info("YOLOv11 Pose Tracking node initialized")

    def depth_callback(self, msg):
        with self.depth_lock:
            try:
                # Convert depth image to numpy array (assuming 16UC1 format)
                self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            except Exception as e:
                self.get_logger().error(f"Depth image conversion error: {str(e)}")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # 使用 YOLO 内置跟踪
            results = self.model.track(
                cv_image,
                persist=True,
                conf=self.conf_threshold,
                tracker="bytetrack.yaml"
            )

            # 直接处理跟踪结果
            tracks = []
            if results and results[0].boxes.id is not None:
                for i in range(len(results[0].boxes)):
                    track = {
                        'track_id': results[0].boxes.id[i].item(),
                        'bbox': results[0].boxes.xyxy[i].cpu().numpy(),
                        'conf': results[0].boxes.conf[i].item(),
                        'keypoints': results[0].keypoints.xy[i].cpu().numpy(),
                        'keypoints_conf': results[0].keypoints.conf[i].cpu().numpy()
                    }
                    tracks.append(track)
            self.print_tracking_info(tracks)

            # 处理每个跟踪的人
            for track in tracks:
                track_id = track['track_id']

                # 初始化或更新跟踪记录
                if track_id not in self.tracked_persons:
                    self.tracked_persons[track_id] = {
                        'is_tracking': False,
                        'hands_up_history': deque(maxlen=30),  # ~1 second at 30fps
                        'hands_up_start_time': None,
                        'hands_up_stop_time': None
                    }

                person = self.tracked_persons[track_id]

                # 检查举手状态
                hands_up = self.is_hands_up(track)
                person['hands_up_history'].append(hands_up)

                # 检查是否举手至少1秒（30帧）
                hands_up_long = sum(person['hands_up_history']) >= 25

                # 跟踪状态机
                if not person['is_tracking']:
                    # 检查是否在冷却期
                    in_cooldown = person['hands_up_stop_time'] is not None and \
                                  (time.time() - person['hands_up_stop_time']) < 10.0

                    if not in_cooldown and hands_up_long:
                        person['is_tracking'] = True
                        person['hands_up_start_time'] = time.time()
                else:
                    # 检查跟踪超时（5秒）
                    if (time.time() - person['hands_up_start_time']) >= 5.0 and hands_up_long:
                        person['is_tracking'] = False
                        person['hands_up_stop_time'] = time.time()

            # 仅过滤活动跟踪
            active_tracks = [t for t in tracks if self.tracked_persons[t['track_id']]['is_tracking']]

            # 可视化和发布结果
            # self.visualize_results(cv_image, tracks)
            self.publish_person_positions(tracks, msg.header)
            # self.publish_keypoints_cloud(active_tracks, msg.header)

            # # # 发布可视化图像
            # detect_pose_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            # detect_pose_msg.header = msg.header
            # self.detect_pose_pub.publish(detect_pose_msg)

        except Exception as e:
            import traceback
            self.get_logger().error(f"Image processing error: {str(e)}\n{traceback.format_exc()}")


    def is_hands_up(self, track: Dict) -> bool:
        keypoints = track['keypoints']

        # 检查关键点是否有效
        if keypoints is None or len(keypoints) < 11:
            return False

        # 关键点索引
        LEFT_WRIST = 9
        RIGHT_WRIST = 10
        LEFT_SHOULDER = 5
        RIGHT_SHOULDER = 6

        # 检查所需关键点是否被检测到
        if (len(keypoints) <= LEFT_WRIST or np.isnan(keypoints[LEFT_WRIST]).any() or
                len(keypoints) <= RIGHT_WRIST or np.isnan(keypoints[RIGHT_WRIST]).any() or
                len(keypoints) <= LEFT_SHOULDER or np.isnan(keypoints[LEFT_SHOULDER]).any() or
                len(keypoints) <= RIGHT_SHOULDER or np.isnan(keypoints[RIGHT_SHOULDER]).any()):
            return False

        # 获取点坐标
        lw = keypoints[LEFT_WRIST]
        rw = keypoints[RIGHT_WRIST]
        ls = keypoints[LEFT_SHOULDER]
        rs = keypoints[RIGHT_SHOULDER]

        # 计算垂直距离
        left_dist = ls[1] - lw[1]  # 肩膀Y - 手腕Y
        right_dist = rs[1] - rw[1]

        # 检查是否有手举起
        return left_dist > 30 or right_dist > 30

    def visualize_results(self, image: np.ndarray, tracks: List[Dict]):
        for track in tracks:
            # Draw bounding box
            x1, y1, x2, y2 = track['bbox'].astype(int)

            # 根据跟踪状态选择颜色
            if self.tracked_persons[track['track_id']]['is_tracking']:
                box_color = (0, 0, 255)  # 红色表示跟踪
                text_color = (0, 0, 255)
            else:

                box_color = (0, 255, 0)  # 绿色表示未跟踪
                text_color = (0, 255, 0)

            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

            # Draw track ID and confidence
            label = f"ID: {track['track_id']} ({track['conf']:.2f})"
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

            # Draw keypoints
            keypoints = track['keypoints']
            for kp in keypoints:
                if not np.isnan(kp).any() and not (kp[0] == 0 and kp[1] == 0):  # 新增(0,0)判断
                    x, y = kp.astype(int)
                    cv2.circle(image, (x, y), 3, (0, 0, 255), -1)

            # Draw skeleton
            for connection in self.skeleton:
                idx1, idx2 = connection
                idx1 -= 1  # 转换为0-based索引
                idx2 -= 1

                if (idx1 < len(keypoints) and idx2 < len(keypoints) and
                        not np.isnan(keypoints[idx1]).any() and not np.isnan(keypoints[idx2]).any() and
                        not (keypoints[idx1][0] == 0 and keypoints[idx1][1] == 0) and  # 新增(0,0)判断
                        not (keypoints[idx2][0] == 0 and keypoints[idx2][1] == 0)):  # 新增(0,0)判断
                    pt1 = keypoints[idx1].astype(int)
                    pt2 = keypoints[idx2].astype(int)
                    cv2.line(image, pt1, pt2, (0, 255, 255), 2)

            # Draw tracking status
            status = "TRACKING" if self.tracked_persons[track['track_id']]['is_tracking'] else "IDLE"
            status_color = (0, 0, 255) if self.tracked_persons[track['track_id']]['is_tracking'] else (255, 0, 0)
            cv2.putText(image, status, (x1, y2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

    def compute_body_depth(self, track: Dict) -> float:
        """计算所有有效关键点的平均深度"""
        valid_depths = []

        with self.depth_lock:
            if self.depth_image is None:
                return 0.0

            keypoints = track['keypoints']
            for kp in keypoints:
                if np.isnan(kp).any() or (kp[0] == 0 and kp[1] == 0):
                    continue

                x, y = kp.astype(int)
                if 0 <= x < self.depth_image.shape[1] and 0 <= y < self.depth_image.shape[0]:
                    depth = self.depth_image[y, x] / 1000.0  # 转换为米
                    if 0.1 < depth < 10.0:  # 有效深度范围
                        valid_depths.append(depth)

        if not valid_depths:
            return 0.0

        return sum(valid_depths) / len(valid_depths)

    def publish_person_positions(self, tracks: List[Dict], header):
        for track in tracks:
            if not self.tracked_persons[track['track_id']]['is_tracking']:
                continue

            # 计算所有有效关键点的平均深度
            avg_depth = self.compute_body_depth(track)
            if avg_depth <= 0.1 or avg_depth >= 10.0:  # 无效深度
                continue

            # 计算边界框中心点
            x1, y1, x2, y2 = track['bbox']
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # 转换为3D相机坐标
            cam_x = (center_x - self.cx) * avg_depth / self.fx
            cam_y = (center_y - self.cy) * avg_depth / self.fy
            cam_z = avg_depth

            # 创建并发布PointStamped消息
            point_msg = PointStamped()
            point_msg.header = header
            point_msg.header.frame_id = "camera_link"
            point_msg.point.x = cam_x
            point_msg.point.y = cam_y
            point_msg.point.z = cam_z

            self.person_point_pub.publish(point_msg)
            return  # 只发布一个活动目标

    def publish_keypoints_cloud(self, tracks: List[Dict], header):
        with self.depth_lock:
            if self.depth_image is None:
                return
            
            # Create PointCloud2 message
            cloud_msg = PointCloud2()
            cloud_msg.header = header
            cloud_msg.header.frame_id = "camera_link"
            
            # Define point fields
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1)
            ]
            
            cloud_msg.fields = fields
            cloud_msg.point_step = 16  # 4 bytes per field * 4 fields
            cloud_msg.height = 1
            cloud_msg.is_dense = True
            
            # Collect valid keypoints
            points = []
            for track in tracks:
                if not self.tracked_persons[track['track_id']]['is_tracking']:
                    continue
                
                # Generate unique color for each track
                track_id = track['track_id']
                r = int((track_id * 50) % 255)
                g = int((track_id * 100) % 255)
                b = int((track_id * 150) % 255)
                rgb = (r << 16) | (g << 8) | b
                
                for kp in track['keypoints']:
                    if np.isnan(kp).any():
                        continue
                    
                    x, y = kp.astype(int)
                    if 0 <= x < self.depth_image.shape[1] and 0 <= y < self.depth_image.shape[0]:
                        depth = self.depth_image[y, x] / 1000.0
                        
                        if depth > 0.1 and depth < 10.0:  # Valid depth range
                            # Convert to 3D camera coordinates
                            cam_x = (x - self.cx) * depth / self.fx
                            cam_y = (y - self.cy) * depth / self.fy
                            cam_z = depth
                            
                            points.append((cam_x, cam_y, cam_z, rgb))
            
            # Set point cloud data
            cloud_msg.width = len(points)
            cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
            
            # Pack data into byte array
            data = bytearray()
            for point in points:
                x, y, z, rgb = point
                data.extend(np.float32(x).tobytes())
                data.extend(np.float32(y).tobytes())
                data.extend(np.float32(z).tobytes())
                data.extend(np.uint32(rgb).tobytes())
            
            cloud_msg.data = bytes(data)
            self.keypoints_cloud_pub.publish(cloud_msg)

    def print_tracking_info(self, tracks: List[Dict]):
        self.get_logger().info("===== Tracking Results =====")
        self.get_logger().info(f"Total tracks: {len(tracks)}")
        
        for track in tracks:
            track_id = track['track_id']
            is_tracking = self.tracked_persons.get(track_id, {}).get('is_tracking', False)
            
            self.get_logger().info(
                f"Track ID: {track_id} | Active: {'YES' if is_tracking else 'NO'} | "
                f"Score: {track['conf']:.2f} | "
                f"BBox: [{track['bbox'][0]:.1f},{track['bbox'][1]:.1f},{track['bbox'][2]:.1f},{track['bbox'][3]:.1f}]"
            )
        
        self.get_logger().info("===========================")

def main(args=None):
    rclpy.init(args=args)
    
    # Use multi-threaded executor for better performance
    executor = MultiThreadedExecutor()
    node = Yolov11PoseNode()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
