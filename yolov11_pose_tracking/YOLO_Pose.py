#!/user/bin/env python

# Copyright (c) 2024，WuChao D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import numpy as np
from scipy.special import softmax
from hobot_dnn import pyeasy_dnn as dnn
import argparse
import logging 

# 日志模块配置
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] [%(levelname)s] %(message)s')
logger = logging.getLogger("RDK_YOLO")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default='models/yolo11n_pose_bayese_640x640_nv12_modified.bin', 
                        help="Path to BPU Quantized *.bin Model")
    parser.add_argument('--test-img', type=str, default='../../../resource/assets/bus.jpg', help='Path to Load Test Image.')
    parser.add_argument('--img-save-path', type=str, default='jupyter_result.jpg', help='Path to Load Test Image.')
    parser.add_argument('--classes-num', type=int, default=80, help='Classes Num to Detect.')
    parser.add_argument('--reg', type=int, default=16, help='DFL reg layer.')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IoU threshold.')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold.')
    parser.add_argument('--kpt-conf-thres', type=float, default=0.5, help='confidence threshold.')
    opt = parser.parse_args()

    # 实例化
    model = YOLO11_Pose(opt.model_path, opt.conf_thres, opt.iou_thres)
    # 读图
    img = cv2.imread(opt.test_img)
    # 准备输入数据
    input_tensor = model.bgr2nv12(img)
    # 推理
    outputs = model.c2numpy(model.forward(input_tensor))
    # 后处理
    ids, scores, bboxes, kpts_xy, kpts_score = model.postProcess(outputs)
    # 渲染
    kpt_conf_inverse = -np.log(1/opt.kpt_conf_thres - 1)
    for class_id, score, bbox, kpt_xy, kpt_score in zip(ids, scores, bboxes, kpts_xy, kpts_score):
        x1, y1, x2, y2 = bbox
        logger.info("(%d, %d, %d, %d) -> %s: %.2f"%(x1,y1,x2,y2, coco_names[class_id], score))
        draw_detection(img, (x1, y1, x2, y2), score, class_id)
        for j in range(17):
            if kpt_score[j,0] < kpt_conf_inverse:
                continue
            x, y = int(kpt_xy[j,0]), int(kpt_xy[j,1])
            cv2.circle(img, (x,y), 5, (0, 0, 255), -1)
            cv2.circle(img, (x,y), 2, (0, 255, 255), -1)
            cv2.putText(img, "%d"%j, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 3, cv2.LINE_AA)
            cv2.putText(img, "%d"%j, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)

    # 保存结果
    cv2.imwrite(opt.img_save_path, img)
    logger.info(f"saved in path: \"./{opt.img_save_path}\"")

class BaseModel:
    def __init__(self, model_file: str) -> None:
        # 加载BPU的bin模型
        try:
            self.quantize_model = dnn.load(model_file)
        except Exception as e:
            logger.error("❌ Failed to load model file: %s"%(model_file))
            logger.error(e)
            exit(1)

        self.model_input_height, self.model_input_weight = self.quantize_model[0].inputs[0].properties.shape[2:4]

    def resizer(self, img: np.ndarray) -> np.ndarray:
        img_h, img_w = img.shape[0:2]
        self.y_scale, self.x_scale = img_h/self.model_input_height, img_w/self.model_input_weight
        return cv2.resize(img, (self.model_input_height, self.model_input_weight), interpolation=cv2.INTER_NEAREST)

    def bgr2nv12(self, bgr_img: np.ndarray) -> np.ndarray:
        """Convert a BGR image to the NV12 format."""
        bgr_img = self.resizer(bgr_img)
        height, width = bgr_img.shape[0], bgr_img.shape[1]
        area = height * width
        yuv420p = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YUV_I420).reshape((area * 3 // 2,))
        y = yuv420p[:area]
        uv_planar = yuv420p[area:].reshape((2, area // 4))
        uv_packed = uv_planar.transpose((1, 0)).reshape((area // 2,))
        nv12 = np.zeros_like(yuv420p)
        nv12[:height * width] = y
        nv12[height * width:] = uv_packed
        return nv12

    def forward(self, input_tensor: np.array) -> list[dnn.pyDNNTensor]:
        return self.quantize_model[0].forward(input_tensor)

    def c2numpy(self, outputs) -> list[np.array]:
        return [dnnTensor.buffer for dnnTensor in outputs]

class YOLO11_Pose(BaseModel):
    def __init__(self, model_file: str, conf: float, iou: float):
        super().__init__(model_file)
        # 反量化系数
        self.s_bboxes_scale = self.quantize_model[0].outputs[0].properties.scale_data[np.newaxis, :]
        self.m_bboxes_scale = self.quantize_model[0].outputs[2].properties.scale_data[np.newaxis, :]
        self.l_bboxes_scale = self.quantize_model[0].outputs[4].properties.scale_data[np.newaxis, :]

        # DFL求期望的系数
        self.weights_static = np.array([i for i in range(16)]).astype(np.float32)[np.newaxis, np.newaxis, :]

        # anchors
        self.s_anchor = np.stack([np.tile(np.linspace(0.5, 79.5, 80), reps=80), 
                            np.repeat(np.arange(0.5, 80.5, 1), 80)], axis=0).transpose(1,0)
        self.m_anchor = np.stack([np.tile(np.linspace(0.5, 39.5, 40), reps=40), 
                            np.repeat(np.arange(0.5, 40.5, 1), 40)], axis=0).transpose(1,0)
        self.l_anchor = np.stack([np.tile(np.linspace(0.5, 19.5, 20), reps=20), 
                            np.repeat(np.arange(0.5, 20.5, 1), 20)], axis=0).transpose(1,0)

        # 输入图像大小和阈值
        self.input_image_size = 640
        self.conf = conf
        self.iou = iou
        self.conf_inverse = -np.log(1/conf - 1)

    def postProcess(self, outputs: list[np.ndarray]) -> tuple[list]:
        # reshape
        s_bboxes = outputs[0].reshape(-1, 64)
        s_clses =  outputs[1].reshape(-1, 1)
        m_bboxes = outputs[2].reshape(-1, 64)
        m_clses =  outputs[3].reshape(-1, 1)
        l_bboxes = outputs[4].reshape(-1, 64)
        l_clses =  outputs[5].reshape(-1, 1)
        s_kpts =   outputs[6].reshape(-1, 51)
        m_kpts =   outputs[7].reshape(-1, 51)
        l_kpts =   outputs[8].reshape(-1, 51)

        # classify: 阈值筛选
        s_max_scores = np.max(s_clses, axis=1)
        s_valid_indices = np.flatnonzero(s_max_scores >= self.conf_inverse)
        s_ids = np.argmax(s_clses[s_valid_indices, : ], axis=1)
        s_scores = s_max_scores[s_valid_indices]

        m_max_scores = np.max(m_clses, axis=1)
        m_valid_indices = np.flatnonzero(m_max_scores >= self.conf_inverse)
        m_ids = np.argmax(m_clses[m_valid_indices, : ], axis=1)
        m_scores = m_max_scores[m_valid_indices]

        l_max_scores = np.max(l_clses, axis=1)
        l_valid_indices = np.flatnonzero(l_max_scores >= self.conf_inverse)
        l_ids = np.argmax(l_clses[l_valid_indices, : ], axis=1)
        l_scores = l_max_scores[l_valid_indices]

        # Sigmoid计算
        s_scores = 1 / (1 + np.exp(-s_scores))
        m_scores = 1 / (1 + np.exp(-m_scores))
        l_scores = 1 / (1 + np.exp(-l_scores))

        # Bounding Box处理
        s_bboxes_float32 = s_bboxes[s_valid_indices,:].astype(np.float32) * self.s_bboxes_scale
        m_bboxes_float32 = m_bboxes[m_valid_indices,:].astype(np.float32) * self.m_bboxes_scale
        l_bboxes_float32 = l_bboxes[l_valid_indices,:].astype(np.float32) * self.l_bboxes_scale

        # dist2bbox (ltrb2xyxy)
        s_ltrb_indices = np.sum(softmax(s_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        s_anchor_indices = self.s_anchor[s_valid_indices, :]
        s_x1y1 = s_anchor_indices - s_ltrb_indices[:, 0:2]
        s_x2y2 = s_anchor_indices + s_ltrb_indices[:, 2:4]
        s_dbboxes = np.hstack([s_x1y1, s_x2y2])*8.0

        m_ltrb_indices = np.sum(softmax(m_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        m_anchor_indices = self.m_anchor[m_valid_indices, :]
        m_x1y1 = m_anchor_indices - m_ltrb_indices[:, 0:2]
        m_x2y2 = m_anchor_indices + m_ltrb_indices[:, 2:4]
        m_dbboxes = np.hstack([m_x1y1, m_x2y2])*16.0

        l_ltrb_indices = np.sum(softmax(l_bboxes_float32.reshape(-1, 4, 16), axis=2) * self.weights_static, axis=2)
        l_anchor_indices = self.l_anchor[l_valid_indices,:]
        l_x1y1 = l_anchor_indices - l_ltrb_indices[:, 0:2]
        l_x2y2 = l_anchor_indices + l_ltrb_indices[:, 2:4]
        l_dbboxes = np.hstack([l_x1y1, l_x2y2])*32.0

        # 关键点处理
        s_kpts = s_kpts[s_valid_indices,:].reshape(-1, 17, 3)
        s_kpts_xy = (s_kpts[:, :, :2] * 2.0 + (self.s_anchor[s_valid_indices,:][:,np.newaxis,:] - 0.5)) * 8.0
        s_kpts_score = s_kpts[:, :, 2:3]

        m_kpts = m_kpts[m_valid_indices,:].reshape(-1, 17, 3)
        m_kpts_xy = (m_kpts[:, :, :2] * 2.0 + (self.m_anchor[m_valid_indices,:][:,np.newaxis,:] - 0.5)) * 16.0
        m_kpts_score = m_kpts[:, :, 2:3]

        l_kpts = l_kpts[l_valid_indices,:].reshape(-1, 17, 3)
        l_kpts_xy = (l_kpts[:, :, :2] * 2.0 + (self.l_anchor[l_valid_indices,:][:,np.newaxis,:] - 0.5)) * 32.0
        l_kpts_score = l_kpts[:, :, 2:3]

        # 特征层结果拼接
        dbboxes = np.concatenate((s_dbboxes, m_dbboxes, l_dbboxes), axis=0)
        scores = np.concatenate((s_scores, m_scores, l_scores), axis=0)
        ids = np.concatenate((s_ids, m_ids, l_ids), axis=0)
        kpts_xy = np.concatenate((s_kpts_xy, m_kpts_xy, l_kpts_xy), axis=0)
        kpts_score  = np.concatenate((s_kpts_score, m_kpts_score, l_kpts_score), axis=0)

        # nms
        indices = cv2.dnn.NMSBoxes(dbboxes, scores, self.conf, self.iou)

        # 还原到原始img尺度
        bboxes = dbboxes[indices] * np.array([self.x_scale, self.y_scale, self.x_scale, self.y_scale])
        bboxes = bboxes.astype(np.int32)
        kpts_xy = kpts_xy[indices] * np.array([[self.x_scale, self.y_scale]])
        kpts_score = kpts_score[indices]

        return ids[indices], scores[indices], bboxes, kpts_xy, kpts_score

coco_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", 
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", 
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", 
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", 
    "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", 
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

rdk_colors = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),(49, 210, 207), (10, 249, 72), (23, 204, 146), (134, 219, 61),
    (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),(147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132),
    (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255)]

def draw_detection(img: np.array, bbox: tuple[int, int, int, int], score: float, class_id: int) -> None:
    x1, y1, x2, y2 = bbox
    color = rdk_colors[class_id%20]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    label = f"{coco_names[class_id]}: {score:.2f}"
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    label_x, label_y = x1, y1 - 10 if y1 - 10 > label_height else y1 + 10
    cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

if __name__ == "__main__":
    main()