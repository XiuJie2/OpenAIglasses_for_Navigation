# -*- coding: utf-8 -*-
"""
YOLOv8 单类分割 + MediaPipe Hand Landmarker + 光流追踪（多边形）
服务器版本 - 专用于ESP32数据源，移除前端和音频功能
提供REST API接口供其他程序获取检测结果
"""

import os
import time
import threading
import math
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import queue
import json
import logging
from typing import Optional, Dict, Any, Tuple, List, Union
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ========= 配置参数 =========
PERF_DEBUG = False
HAND_DOWNSCALE = 0.8
HAND_FPS_DIV = 1

# 输入源配置 - 强制使用ESP32
INPUT_SOURCE = "esp32"
INPUT_W, INPUT_H = 600, 480

# 模型路径
HAND_TASK_PATH = r"model\hand_landmarker.task"

# 分割参数
STROKE_WIDTH = 5
MASK_ALPHA = 0.45
CONF_THRESHOLD = 0.20
PROMPT_NAME = "AD_milk"

# 其他参数
ALIGN_LOOSE_PCT = 0.12
RATIO_IDEAL = 1.0
RATIO_TOL = 0.25
INNER_OFFSET_PX_LOCK = 5
EDGE_DILATE_PX = 2
PERI_MONITOR_PX = 40
PERI_CHECK_EVERY = 5
CONTOUR_EPSILON_FACTOR = 0.002
TRACK_EPSILON_FACTOR = 0.003
YOLO_CORRECTION_IOU_THRESHOLD = 0.2
YOLO_CORRECTION_CONF_THRESHOLD = 0.15

# ========= MediaPipe 配置 =========
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

# 尝试导入YOLOE后端
try:
    from yoloe_backend import YoloEBackend
    _YOLOE_READY = True
except Exception as e:
    _YOLOE_READY = False
    logger.warning(f"YOLOE backend not ready: {e}")

# ========= 结果队列（用于API访问）=========
class ResultQueue:
    def __init__(self, max_size=100):
        self.queue = queue.Queue(maxsize=max_size)
        self.latest_result = None
        self.lock = threading.Lock()
        self.stats = {
            "total_frames": 0,
            "processed_frames": 0,
            "start_time": time.time(),
            "last_update": time.time()
        }
    
    def put(self, result: Dict[str, Any]):
        """放入结果"""
        with self.lock:
            self.latest_result = result
            self.stats["total_frames"] += 1
            self.stats["processed_frames"] += 1
            self.stats["last_update"] = time.time()
            
            try:
                self.queue.put_nowait(result)
            except queue.Full:
                # 如果队列满了，移除最旧的结果
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
                self.queue.put_nowait(result)
    
    def get_latest(self) -> Optional[Dict[str, Any]]:
        """获取最新结果"""
        with self.lock:
            return self.latest_result
    
    def get_all(self, max_items: int = 10) -> List[Dict[str, Any]]:
        """获取最近的结果"""
        results = []
        with self.lock:
            # 清空队列并获取所有结果
            while not self.queue.empty():
                try:
                    results.append(self.queue.get_nowait())
                except queue.Empty:
                    break
        
        # 只返回最新的几个结果
        return results[-max_items:]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self.lock:
            elapsed = time.time() - self.stats["start_time"]
            fps = self.stats["processed_frames"] / max(elapsed, 0.001)
            
            return {
                "total_frames": self.stats["total_frames"],
                "processed_frames": self.stats["processed_frames"],
                "fps": round(fps, 2),
                "uptime_seconds": round(elapsed, 2),
                "last_update": datetime.fromtimestamp(self.stats["last_update"]).isoformat(),
                "queue_size": self.queue.qsize()
            }

# 全局结果队列
result_queue = ResultQueue()

# ======== HandLandmarker 回调缓存 ========
_last_result = None  # (result, timestamp_ms)

def on_result(result: mp.tasks.vision.HandLandmarkerResult,
              output_image: mp.Image, timestamp_ms: int):
    global _last_result
    _last_result = (result, timestamp_ms)

def _to_proto(hand_lms) -> landmark_pb2.NormalizedLandmarkList:
    proto = landmark_pb2.NormalizedLandmarkList()
    proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=p.x, y=p.y, z=p.z) for p in hand_lms
    ])
    return proto

# ======== 工具函数 ========
def clamp01(x): return max(0.0, min(1.0, x))

def polygon_center_and_area(poly):
    if poly is None or len(poly) < 3:
        return None, 0.0
    poly = np.array(poly, dtype=np.float32)
    M = cv2.moments(poly)
    if abs(M["m00"]) < 1e-6:
        c = np.mean(poly, axis=0)
        return (float(c[0]), float(c[1])), 0.0
    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])
    area = float(cv2.contourArea(poly.astype(np.int32)))
    return (cx, cy), area

def hand_bbox_and_area(lms, W, H):
    xs = [int(p.x * W) for p in lms]
    ys = [int(p.y * H) for p in lms]
    if not xs or not ys:
        return None, 0.0
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    w = max(1, x1 - x0)
    h = max(1, y1 - y0)
    area = float(w * h)
    return (x0, y0, w, h), area

# ======== 手势：握持(Grasp) 识别（放宽版启发式）========
THUMB_INDEX_CLOSE = 0.34   # 放宽
FINGERTIP_NEAR = 0.44   # 放宽
MIN_CURLED_COUNT = 1      # 放宽

def detect_grasp(hand_lms, W, H):
    box, _ = hand_bbox_and_area(hand_lms, W, H)
    if not box:
        return False, 0.0
    x0, y0, w0, h0 = box
    hand_diag = float(np.hypot(w0, h0)) + 1e-6
    palm_idx = [0, 5, 9, 13, 17]
    px = np.mean([hand_lms[i].x * W for i in palm_idx])
    py = np.mean([hand_lms[i].y * H for i in palm_idx])
    palm = np.array([px, py], dtype=np.float32)
    t4 = np.array([hand_lms[4].x * W, hand_lms[4].y * H], dtype=np.float32)
    t8 = np.array([hand_lms[8].x * W, hand_lms[8].y * H], dtype=np.float32)
    thumb_index_dist = float(np.linalg.norm(t4 - t8)) / hand_diag
    tips = [12, 16, 20]
    dists = []
    for i in tips:
        ti = np.array([hand_lms[i].x * W, hand_lms[i].y * H], dtype=np.float32)
        dists.append(float(np.linalg.norm(ti - palm)) / hand_diag)
    curled_cnt = sum(1 for d in dists if d < FINGERTIP_NEAR)
    cond1 = (thumb_index_dist < THUMB_INDEX_CLOSE)
    cond2 = (curled_cnt >= MIN_CURLED_COUNT)
    score = 0.5 * (1.0 - min(thumb_index_dist / THUMB_INDEX_CLOSE, 1.0)) + \
            0.5 * min(curled_cnt / 3.0, 1.0)
    return (cond1 and cond2), score

# ======== 内收后的边界提点 ========
def inner_offset_edge(mask_bin, offset_px=5, edge_dilate_px=2):
    if offset_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*offset_px+1, 2*offset_px+1))
        eroded = cv2.erode(mask_bin.astype(np.uint8), k, iterations=1)
    else:
        eroded = mask_bin.astype(np.uint8)
    edges = cv2.Canny(eroded*255, 50, 150)
    if edge_dilate_px > 0:
        k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*edge_dilate_px+1, 2*edge_dilate_px+1))
        edges = cv2.dilate(edges, k2, iterations=1)
    return edges  # uint8 0/255

# 检测手和物体是否接触
def check_hand_object_contact(hand_box, poly, overlap_threshold=0.15):
    """
    检测手的边界框和物体多边形是否有重叠
    返回: (是否接触, 重叠比例)
    """
    if hand_box is None or poly is None or len(poly) < 3:
        return False, 0.0
    
    # 获取手的边界框
    hx, hy, hw, hh = hand_box
    hand_rect = np.array([
        [hx, hy],
        [hx + hw, hy],
        [hx + hw, hy + hh],
        [hx, hy + hh]
    ], dtype=np.int32)
    
    # 创建掩码来计算重叠
    H = int(max(hy + hh, np.max(poly[:, 1])) + 10)
    W = int(max(hx + hw, np.max(poly[:, 0])) + 10)
    
    hand_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(hand_mask, [hand_rect], 1)
    
    obj_mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(obj_mask, [poly.astype(np.int32)], 1)
    
    # 计算重叠
    intersection = np.logical_and(hand_mask, obj_mask).sum()
    hand_area = hand_mask.sum()
    
    # 重叠比例（相对于手的面积）
    overlap_ratio = intersection / max(1.0, hand_area)
    
    return overlap_ratio > overlap_threshold, overlap_ratio

# 添加方向判断函数
def get_guidance_direction(hand_center, object_center, hand_area, object_area, hand_box=None, poly=None):
    """
    根据手心和物体中心位置，以及面积比，返回引导方向
    返回: (方向文字, 是否需要前后调整)
    """
    if hand_center is None or object_center is None:
        return None, None
    
    # 首先检查手和物体是否接触
    is_touching = False
    overlap_ratio = 0.0
    if hand_box is not None and poly is not None:
        is_touching, overlap_ratio = check_hand_object_contact(hand_box, poly, overlap_threshold=0.1)
    
    hx, hy = hand_center
    ox, oy = object_center
    
    # 计算水平和垂直偏差
    dx = ox - hx  # 正数表示物体在右边
    dy = oy - hy  # 正数表示物体在下边
    
    # 如果手和物体已经接触，直接返回"向前"
    if is_touching:
        return "向前", f"接触度: {overlap_ratio:.1%}"
    
    # 如果没有接触，引导上下左右
    # 判断主要方向
    h_threshold = 30  # 水平偏差阈值（像素）
    v_threshold = 30  # 垂直偏差阈值（像素）
    
    h_dir = None
    v_dir = None
    
    # 水平方向
    if abs(dx) > h_threshold:
        h_dir = "向右" if dx > 0 else "向左"
    
    # 垂直方向
    if abs(dy) > v_threshold:
        v_dir = "向下" if dy > 0 else "向上"
    
    # 选择偏移最大的方向
    if abs(dx) > abs(dy) and h_dir:
        # 水平偏移更大
        return h_dir, v_dir
    elif v_dir:
        # 垂直偏移更大或相等
        return v_dir, h_dir
    else:
        # 已经在中心附近但还没接触，提示靠近
        distance = np.sqrt(dx**2 + dy**2)
        if distance < 50:  # 很近但还没接触
            return "向前", "请缓慢靠近"
        else:
            return "保持", None

# 添加居中判断函数
def get_center_guidance(object_center, frame_center, threshold=30):
    """
    判断物体是否在画面中心，返回引导方向
    返回: (方向文字, 是否已居中)
    """
    if object_center is None:
        return None, False
    
    ox, oy = object_center
    cx, cy = frame_center
    
    dx = cx - ox  # 正数表示需要向右移动
    dy = cy - oy  # 正数表示需要向下移动
    
    # 判断是否已经居中
    distance = np.sqrt(dx**2 + dy**2)
    if distance < threshold:
        return "已居中", True
    
    # 判断主要方向（对调左右和上下）
    if abs(dx) > abs(dy):
        return "向左" if dx > 0 else "向右", False  # 对调了
    else:
        return "向上" if dy > 0 else "向下", False  # 对调了

# ========= 光流（LK）与特征点 =========
LK_PARAMS = dict(winSize=(21, 21),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03))
FEATURE_PARAMS = dict(maxCorners=600,
                      qualityLevel=0.001,
                      minDistance=5,
                      blockSize=7)

class ObjectSearchServer:
    """
    物品查找服务器 - 处理ESP32视频流，提供REST API接口
    """
    
    def __init__(self, prompt_name: str = None, config: Dict[str, Any] = None):
        """
        初始化服务器
        
        Args:
            prompt_name: 目标物体提示词
            config: 配置参数
        """
        # 配置参数
        self.config = config or {}
        self.prompt_name = prompt_name or PROMPT_NAME
        
        # 状态变量
        self.running = False
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        # 处理状态
        self.MODE = "SEGMENT"
        self.FRAME_IDX = 0
        self.last_mask = None
        self.flow_mask = None
        self.flow_grace = 0
        self.last_seen_ts = 0.0
        self.locked_id = None
        
        # 自动锁定相关
        self.auto_lock_start_time = None
        self.auto_lock_delay = 1.0
        self.last_detected_mask = None
        
        # 闪烁动画相关
        self.flash_start_time = None
        self.flash_duration = 1.0
        self.flash_mask = None
        
        # 居中引导相关
        self.center_guide_start = None
        self.center_reached = False
        self.last_center_guide_time = 0
        
        # 光流缓存
        self.old_gray = None
        self.p0 = None
        self.track_frame_count = 0
        self.last_poly_box = None
        
        # 背景参考点
        self.background_points = None
        self.old_background_gray = None
        
        # 引导相关
        self.last_guidance_time = 0
        self.last_guidance_direction = None
        
        # 模型后端
        self.use_yoloe = False
        self.yoloe_backend = None
        self.landmarker = None
        
        # FPS计算
        self.fps_hist = []
        
        logger.info(f"ObjectSearchServer initialized with prompt: {self.prompt_name}")
    
    def initialize_models(self):
        """初始化模型"""
        try:
            # 初始化YOLOE后端
            if _YOLOE_READY:
                self.yoloe_backend = YoloEBackend()
                self.yoloe_backend.set_text_classes([self.prompt_name])
                self.use_yoloe = True
                logger.info(f"YOLOE text-prompt backend enabled for: {self.prompt_name}")
            else:
                logger.warning("YOLOE backend not available")
            
            # 初始化MediaPipe Hand Landmarker
            base = BaseOptions(model_asset_path=HAND_TASK_PATH)
            hand_options = HandLandmarkerOptions(
                base_options=base,
                running_mode=VisionRunningMode.LIVE_STREAM,
                num_hands=1,
                min_hand_detection_confidence=0.40,
                min_hand_presence_confidence=0.50,
                min_tracking_confidence=0.70,
                result_callback=on_result
            )
            self.landmarker = HandLandmarker.create_from_options(hand_options)
            
            logger.info("Models initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        处理一帧图像
        
        Args:
            frame: 输入图像帧
            
        Returns:
            处理结果字典
        """
        # 调整帧大小到目标分辨率
        if frame.shape[1] != INPUT_W or frame.shape[0] != INPUT_H:
            frame = cv2.resize(frame, (INPUT_W, INPUT_H))
        
        H, W = frame.shape[:2]
        t_now = time.time()
        
        # 抽帧 + 降采样（人手识别）
        if self.FRAME_IDX % HAND_FPS_DIV == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if HAND_DOWNSCALE and HAND_DOWNSCALE != 1.0:
                small = cv2.resize(rgb, None, fx=HAND_DOWNSCALE, fy=HAND_DOWNSCALE, interpolation=cv2.INTER_AREA)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=small)
            else:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self.landmarker.detect_async(mp_image, int(t_now * 1000))
        
        # 收集检测结果
        detection_result = {
            "timestamp": t_now,
            "frame_id": self.FRAME_IDX,
            "mode": self.MODE,
            "hand_detected": False,
            "object_detected": False,
            "guidance": None,
            "metrics": {},
            "polygon": None,
            "status": "idle",
            "processing_time_ms": 0
        }
        
        start_time = time.time()
        
        # 取手心、手框、握持（放宽版）
        hand_center = None
        hand_area = None
        hand_box = None
        grasp_now = False
        grasp_score = 0.0
        
        if _last_result is not None:
            res, _ = _last_result
            if res.hand_landmarks and len(res.hand_landmarks) > 0:
                l0 = res.hand_landmarks[0]
                
                # 获取手部信息
                xs = [p.x * W for p in l0]
                ys = [p.y * H for p in l0]
                hand_center = (float(sum(xs)/len(xs)), float(sum(ys)/len(ys)))
                hand_box, hand_area = hand_bbox_and_area(l0, W, H)
                grasp_now, grasp_score = detect_grasp(l0, W, H)
                
                detection_result["hand_detected"] = True
                detection_result["hand_center"] = hand_center
                detection_result["hand_box"] = hand_box
                detection_result["grasp_score"] = grasp_score
                detection_result["grasp_now"] = grasp_now
        
        if self.MODE == "SEGMENT":
            self.FRAME_IDX += 1
            candidate_masks = []
            detected_object = False

            if self.use_yoloe and self.yoloe_backend is not None:
                det = self.yoloe_backend.segment(frame, conf=0.20, iou=0.45, imgsz=640, persist=True)
                H, W = frame.shape[:2]

                # 选一个掩膜：优先与 locked_id 相同；否则面积最大
                chosen_idx = None
                if det["masks"]:
                    if self.locked_id is not None and det["ids"] and (self.locked_id in det["ids"]):
                        chosen_idx = det["ids"].index(self.locked_id)
                    else:
                        areas = [int(m.sum()) for m in det["masks"]]
                        chosen_idx = int(np.argmax(areas))

                    if chosen_idx is not None:
                        m = det["masks"][chosen_idx]
                        if m.shape[:2] != (H, W):
                            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

                        mask_bin = (m > 0).astype(np.uint8)
                        candidate_masks.append({
                            "mask": mask_bin,
                            "area": int(mask_bin.sum()),
                            "name": self.prompt_name,
                            "cls_id": 0,
                            "conf": 0.99,
                        })
                        detected_object = True

                        # 记录 id，减少目标跳变
                        if det["ids"] and len(det["ids"]) > chosen_idx and det["ids"][chosen_idx] is not None:
                            self.locked_id = int(det["ids"][chosen_idx])

            # 选择面积最大的mask
            if candidate_masks:
                candidate_masks.sort(key=lambda x: x['area'], reverse=True)
                largest_mask_info = candidate_masks[0]
                self.last_detected_mask = largest_mask_info['mask']
                
                detection_result["object_detected"] = True
                detection_result["object_area"] = int(np.sum(self.last_detected_mask))
                detection_result["object_count"] = len(candidate_masks)
                detection_result["object_confidence"] = largest_mask_info["conf"]
            
            # 自动锁定逻辑
            if detected_object and self.last_detected_mask is not None:
                if self.auto_lock_start_time is None:
                    self.auto_lock_start_time = t_now
                    detection_result["status"] = "detected"
                    detection_result["guidance"] = "检测到物体，准备锁定"
                else:
                    elapsed = t_now - self.auto_lock_start_time
                    remaining = self.auto_lock_delay - elapsed
                    
                    if remaining > 0:
                        detection_result["status"] = f"locking_{remaining:.1f}s"
                        detection_result["guidance"] = f"检测到物体，{remaining:.1f}秒后自动锁定"
                    else:
                        # 进入闪烁模式
                        logger.info("Entering flash animation mode")
                        self.MODE = "FLASH"
                        self.flash_start_time = t_now
                        self.flash_mask = self.last_detected_mask.copy()
                        self.auto_lock_start_time = None
                        
                        detection_result["status"] = "flashing"
                        detection_result["guidance"] = "检测到物体，正在锁定"
            else:
                # 没有检测到物体，重置计时器
                if self.auto_lock_start_time is not None:
                    logger.info("Object lost, resetting countdown")
                self.auto_lock_start_time = None
                self.last_detected_mask = None
                detection_result["status"] = "searching"
                detection_result["guidance"] = "正在搜索物体..."

        elif self.MODE == "FLASH":
            # 闪烁动画模式
            if self.flash_start_time is not None and self.flash_mask is not None:
                elapsed = t_now - self.flash_start_time
                
                if elapsed < self.flash_duration:
                    detection_result["status"] = f"flashing_{elapsed:.1f}s"
                    detection_result["guidance"] = "正在锁定目标..."
                else:
                    # 闪烁结束，初始化光流追踪并进入居中引导模式
                    logger.info("Flash ended, initializing optical flow")
                    edge_mask = inner_offset_edge(self.flash_mask, offset_px=INNER_OFFSET_PX_LOCK, edge_dilate_px=EDGE_DILATE_PX)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    pts = cv2.goodFeaturesToTrack(gray, mask=edge_mask, **FEATURE_PARAMS)
                    
                    if pts is not None and len(pts) >= 8:
                        self.p0 = pts
                        self.old_gray = gray
                        self.MODE = "CENTER_GUIDE"
                        self.track_frame_count = 0
                        self.center_guide_start = t_now
                        self.center_reached = False
                        self.flash_start_time = None
                        self.flash_mask = None
                        self.last_detected_mask = None
                        logger.info(f"Edge feature points={len(self.p0)} → CENTER_GUIDE")
                    else:
                        logger.info("Edge feature points insufficient, returning to detection mode")
                        self.MODE = "SEGMENT"
                        self.flash_start_time = None
                        self.flash_mask = None
                        self.last_detected_mask = None
        
        elif self.MODE == "CENTER_GUIDE":
            # 居中引导模式（使用光流追踪）
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            poly_center = None
            poly_area = 0.0
            
            if self.old_gray is not None and self.p0 is not None and len(self.p0) >= 5:
                # 光流追踪
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, gray, self.p0, None, **LK_PARAMS)
                if p1 is not None and st is not None:
                    good_new = p1[st == 1]
                    if len(good_new) >= 5:
                        self.p0 = good_new.reshape(-1, 1, 2)
                        hull = cv2.convexHull(good_new.reshape(-1,1,2))
                        poly = hull.reshape(-1, 2)
                        
                        if len(poly) >= 3:
                            # 多边形质心与面积
                            poly_center, poly_area = polygon_center_and_area(poly)
                            
                            if poly_center:
                                object_center = (int(poly_center[0]), int(poly_center[1]))
                                frame_center = (W // 2, H // 2)
                                
                                # 获取引导方向
                                direction, is_centered = get_center_guidance(object_center, frame_center, 30)
                                
                                detection_result["object_detected"] = True
                                detection_result["object_center"] = poly_center
                                detection_result["object_area"] = poly_area
                                detection_result["polygon"] = poly.tolist()
                                detection_result["frame_center"] = frame_center
                                
                                if not self.center_reached:
                                    if is_centered:
                                        # 到达中心
                                        self.center_reached = True
                                        self.last_center_guide_time = t_now
                                        detection_result["status"] = "centered"
                                        detection_result["guidance"] = "物品已居中！"
                                    else:
                                        # 显示引导文字
                                        detection_result["status"] = "centering"
                                        detection_result["guidance"] = f"请将物品移到画面中心: {direction}"
                                        
                                        # 计算距离信息
                                        dx = frame_center[0] - object_center[0]
                                        dy = frame_center[1] - object_center[1]
                                        distance = int(np.sqrt(dx**2 + dy**2))
                                        detection_result["metrics"]["distance_to_center"] = distance
                                        
                                        if t_now - self.last_center_guide_time > 1.5:
                                            self.last_center_guide_time = t_now
                                else:
                                    # 已经居中
                                    detection_result["status"] = "centered_holding"
                                    detection_result["guidance"] = "物品已成功移到中心！"
                                    
                                    # 等待1秒后进入手部追踪模式
                                    if t_now - self.last_center_guide_time > 1.0:
                                        logger.info("Entering hand tracking mode")
                                        self.MODE = "TRACK"
                            else:
                                detection_result["status"] = "tracking_object"
                                detection_result["guidance"] = "正在追踪物体..."
                else:
                    # 光流点数不足，尝试重新检测
                    self.MODE = "SEGMENT"
                    self.old_gray = None
                    self.p0 = None
                    logger.info("Optical flow tracking failed, returning to detection mode")
            
            self.old_gray = gray

        elif self.MODE == "TRACK":
            # 手部追踪模式
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.track_frame_count += 1

            relock_done = False
            poly_center = None
            poly_area = 0.0

            if self.old_gray is not None and self.p0 is not None and len(self.p0) >= 5:
                p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, gray, self.p0, None, **LK_PARAMS)
                if p1 is not None and st is not None:
                    good_new = p1[st == 1]
                    if len(good_new) >= 5:
                        self.p0 = good_new.reshape(-1, 1, 2)
                        hull = cv2.convexHull(good_new.reshape(-1,1,2))
                        poly = hull.reshape(-1, 2)
                        
                        if len(poly) >= 3:
                            # 统一的 YOLOE 实时检测和校正（每帧）
                            latest_det_mask = None
                            if self.use_yoloe and self.yoloe_backend is not None:
                                det = self.yoloe_backend.segment(frame, conf=YOLO_CORRECTION_CONF_THRESHOLD, iou=0.45, imgsz=640, persist=True)
                                if det["masks"]:
                                    areas = [int(m.sum()) for m in det["masks"]]
                                    j = int(np.argmax(areas))
                                    m = det["masks"][j]
                                    if m.shape[:2] != (H, W):
                                        m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                                    latest_det_mask = (m > 0).astype(np.uint8)
                                    
                                    # 和当前光流多边形的 IoU
                                    poly_mask = np.zeros((H, W), dtype=np.uint8)
                                    cv2.fillPoly(poly_mask, [poly.astype(np.int32)], 1)
                                    inter = np.logical_and(latest_det_mask, poly_mask).sum()
                                    union = np.logical_or(latest_det_mask, poly_mask).sum() + 1e-6
                                    iou = inter / union
                                    
                                    # 降低IoU阈值，更积极地校正
                                    if iou > YOLO_CORRECTION_IOU_THRESHOLD:
                                        contours, _ = cv2.findContours(latest_det_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                                        if contours:
                                            largest_contour = max(contours, key=cv2.contourArea)
                                            epsilon = TRACK_EPSILON_FACTOR * cv2.arcLength(largest_contour, True)
                                            poly = cv2.approxPolyDP(largest_contour, epsilon, True).reshape(-1, 2)
                                            
                                            edge_mask = inner_offset_edge(latest_det_mask, offset_px=INNER_OFFSET_PX_LOCK, edge_dilate_px=EDGE_DILATE_PX)
                                            pts = cv2.goodFeaturesToTrack(gray, mask=edge_mask, **FEATURE_PARAMS)
                                            if pts is not None and len(pts) >= 5:
                                                self.p0 = pts
                            
                            # 检查是否接触
                            is_touching = False
                            overlap_ratio = 0.0
                            if hand_box is not None and poly is not None:
                                is_touching, overlap_ratio = check_hand_object_contact(hand_box, poly, overlap_threshold=0.1)
                            
                            # 多边形质心与面积
                            poly_center, poly_area = polygon_center_and_area(poly)
                            if poly_center:
                                # 多边形外接矩形
                                x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
                                self.last_poly_box = (x, y, w, h)

                                detection_result["object_detected"] = True
                                detection_result["object_center"] = poly_center
                                detection_result["object_area"] = poly_area
                                detection_result["polygon"] = poly.tolist()
                                detection_result["is_touching"] = is_touching
                                detection_result["overlap_ratio"] = overlap_ratio

                                # 对齐分数
                                if hand_center and poly_center:
                                    hc = np.array(hand_center, dtype=np.float32)
                                    oc = np.array(poly_center, dtype=np.float32)
                                    dist = float(np.linalg.norm(oc - hc))
                                    diag = float(np.linalg.norm([W, H]))
                                    align_score = 1.0 - min(dist/(ALIGN_LOOSE_PCT*diag + 1e-6), 1.0)
                                    
                                    detection_result["metrics"]["hand_object_distance"] = dist
                                    detection_result["metrics"]["align_score"] = align_score
                                    
                                    # 方向引导
                                    direction, secondary = get_guidance_direction(
                                        hand_center, poly_center, hand_area, poly_area,
                                        hand_box, poly
                                    )
                                    
                                    if direction and direction != "保持":
                                        detection_result["guidance"] = direction
                                        detection_result["status"] = "guiding"
                                        
                                        if t_now - self.last_guidance_time > 1.5:
                                            if direction != self.last_guidance_direction or t_now - self.last_guidance_time > 3.0:
                                                self.last_guidance_direction = direction
                                                self.last_guidance_time = t_now
                                                logger.info(f"Guidance: {direction}")
                                    else:
                                        detection_result["status"] = "tracking"
                                
                                # 成功条件：握持（放宽）
                                detection_result["grasp_now"] = grasp_now
                                detection_result["grasp_score"] = grasp_score

                        else:
                            self.MODE = "SEGMENT"
                            self.old_gray = None
                            self.p0 = None
                    else:
                        self.MODE = "SEGMENT"
                        self.old_gray = None
                        self.p0 = None
                else:
                    self.MODE = "SEGMENT"
                    self.old_gray = None
                    self.p0 = None
            else:
                self.MODE = "SEGMENT"
                self.old_gray = None
                self.p0 = None

            if self.MODE == "SEGMENT":
                detection_result["status"] = "tracking_lost"
                detection_result["guidance"] = "追踪丢失 → 正在重新识别"

            self.old_gray = gray
        
        # FPS计算
        self.fps_hist.append(t_now)
        if len(self.fps_hist) > 30:
            self.fps_hist.pop(0)
        fps = 0.0 if len(self.fps_hist) < 2 else (len(self.fps_hist)-1)/(self.fps_hist[-1]-self.fps_hist[0])
        
        detection_result["fps"] = round(fps, 2)
        detection_result["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
        
        return detection_result
    
    def process_stream(self):
        """处理ESP32视频流"""
        try:
            import bridge_io
        except ImportError as e:
            logger.error(f"Failed to import bridge_io: {e}")
            logger.error("Please ensure bridge_io is available for ESP32 communication")
            return
        
        logger.info("Starting ESP32 video stream processing")
        
        while not self.stop_event.is_set():
            try:
                # 从ESP32获取帧
                frame = bridge_io.wait_raw_bgr(timeout_sec=0.5)
                if frame is None:
                    continue
                
                # 处理帧
                result = self.process_frame(frame)
                
                # 将结果放入队列
                result_queue.put(result)
                
                # 可选的：发送可视化帧到ESP32（如果bridge_io支持）
                try:
                    # 这里可以添加生成可视化帧的逻辑，如果需要的话
                    # vis_frame = self.generate_visualization(frame, result)
                    # bridge_io.send_vis_bgr(vis_frame)
                    pass
                except Exception as e:
                    logger.warning(f"Failed to send visualization frame: {e}")
                
                # 发送UI消息到ESP32
                try:
                    if result["guidance"]:
                        bridge_io.send_ui_final(result["guidance"])
                except Exception as e:
                    logger.warning(f"Failed to send UI message: {e}")
                
                # 轻微延迟，避免占用过多CPU
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                time.sleep(0.1)
        
        logger.info("Video stream processing stopped")
    
    def start(self):
        """启动服务器"""
        if self.running:
            logger.warning("Server is already running")
            return False
        
        # 初始化模型
        if not self.initialize_models():
            return False
        
        # 重置停止事件
        self.stop_event.clear()
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self.process_stream)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.running = True
        logger.info("ObjectSearchServer started successfully")
        return True
    
    def stop(self):
        """停止服务器"""
        if not self.running:
            return
        
        logger.info("Stopping ObjectSearchServer...")
        
        # 设置停止事件
        self.stop_event.set()
        
        # 等待处理线程结束
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
        
        # 关闭模型
        if self.landmarker:
            try:
                self.landmarker.close()
                logger.info("Hand landmarker closed")
            except Exception as e:
                logger.error(f"Error closing landmarker: {e}")
        
        self.running = False
        logger.info("ObjectSearchServer stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """获取服务器状态"""
        return {
            "running": self.running,
            "mode": self.MODE,
            "prompt_name": self.prompt_name,
            "frame_count": self.FRAME_IDX,
            "stats": result_queue.get_stats()
        }
    
    def update_prompt(self, prompt_name: str) -> bool:
        """更新目标提示词"""
        try:
            self.prompt_name = prompt_name
            
            if self.use_yoloe and self.yoloe_backend is not None:
                self.yoloe_backend.set_text_classes([prompt_name])
            
            logger.info(f"Prompt name updated to: {prompt_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to update prompt: {e}")
            return False
    
    def reset_state(self):
        """重置服务器状态"""
        logger.info("Resetting server state")
        
        # 重置所有状态变量
        self.MODE = "SEGMENT"
        self.last_mask = None
        self.flow_mask = None
        self.flow_grace = 0
        self.last_seen_ts = 0.0
        self.locked_id = None
        
        self.auto_lock_start_time = None
        self.last_detected_mask = None
        
        self.flash_start_time = None
        self.flash_mask = None
        
        self.center_guide_start = None
        self.center_reached = False
        self.last_center_guide_time = 0
        
        self.old_gray = None
        self.p0 = None
        self.track_frame_count = 0
        self.last_poly_box = None
        
        self.background_points = None
        self.old_background_gray = None
        
        self.last_guidance_time = 0
        self.last_guidance_direction = None
        
        logger.info("Server state reset completed")


# ========= REST API 接口 =========
from flask import Flask, jsonify, request, Response
import threading

app = Flask(__name__)
server_instance = None

@app.route('/api/status', methods=['GET'])
def get_status():
    """获取服务器状态"""
    if server_instance is None:
        return jsonify({"error": "Server not initialized"}), 500
    
    status = server_instance.get_status()
    return jsonify(status)

@app.route('/api/latest', methods=['GET'])
def get_latest_result():
    """获取最新的检测结果"""
    result = result_queue.get_latest()
    if result:
        return jsonify(result)
    else:
        return jsonify({"error": "No results available"}), 404

@app.route('/api/results', methods=['GET'])
def get_recent_results():
    """获取最近的检测结果"""
    max_items = request.args.get('max', default=10, type=int)
    results = result_queue.get_all(max_items)
    return jsonify({
        "count": len(results),
        "results": results
    })

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """获取统计信息"""
    stats = result_queue.get_stats()
    return jsonify(stats)

@app.route('/api/update_prompt', methods=['POST'])
def update_prompt():
    """更新目标提示词"""
    if server_instance is None:
        return jsonify({"error": "Server not initialized"}), 500
    
    data = request.json
    if not data or 'prompt_name' not in data:
        return jsonify({"error": "prompt_name is required"}), 400
    
    success = server_instance.update_prompt(data['prompt_name'])
    
    if success:
        return jsonify({
            "success": True,
            "prompt_name": data['prompt_name']
        })
    else:
        return jsonify({"error": "Failed to update prompt"}), 500

@app.route('/api/control/start', methods=['POST'])
def start_server():
    """启动服务器"""
    global server_instance
    
    if server_instance is None:
        # 从请求中获取配置
        data = request.json or {}
        prompt_name = data.get('prompt_name', PROMPT_NAME)
        
        # 创建服务器实例
        server_instance = ObjectSearchServer(prompt_name=prompt_name)
    
    if server_instance.start():
        return jsonify({"success": True, "message": "Server started"})
    else:
        return jsonify({"error": "Failed to start server"}), 500

@app.route('/api/control/stop', methods=['POST'])
def stop_server():
    """停止服务器"""
    global server_instance
    
    if server_instance is None:
        return jsonify({"error": "Server not initialized"}), 400
    
    server_instance.stop()
    return jsonify({"success": True, "message": "Server stopped"})

@app.route('/api/control/reset', methods=['POST'])
def reset_server():
    """重置服务器状态"""
    global server_instance
    
    if server_instance is None:
        return jsonify({"error": "Server not initialized"}), 400
    
    server_instance.reset_state()
    return jsonify({"success": True, "message": "Server state reset"})

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    if server_instance is None or not server_instance.running:
        return jsonify({"status": "not_running"}), 503
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })

@app.route('/')
def index():
    """API文档页面"""
    return """
    <h1>Object Search Server API</h1>
    <h2>Endpoints:</h2>
    <ul>
        <li>GET /api/status - 获取服务器状态</li>
        <li>GET /api/latest - 获取最新检测结果</li>
        <li>GET /api/results?max=10 - 获取最近的结果</li>
        <li>GET /api/stats - 获取统计信息</li>
        <li>POST /api/update_prompt - 更新目标提示词</li>
        <li>POST /api/control/start - 启动服务器</li>
        <li>POST /api/control/stop - 停止服务器</li>
        <li>POST /api/control/reset - 重置服务器状态</li>
        <li>GET /api/health - 健康检查</li>
    </ul>
    """

def run_api_server(host='0.0.0.0', port=5000):
    """启动API服务器"""
    logger.info(f"Starting API server on {host}:{port}")
    app.run(host=host, port=port, threaded=True)


# ========= 主函数 =========
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Object Search Server for ESP32')
    parser.add_argument('--prompt', type=str, default=PROMPT_NAME, help='Prompt name for object detection')
    parser.add_argument('--api-port', type=int, default=5000, help='Port for REST API server')
    parser.add_argument('--api-host', type=str, default='0.0.0.0', help='Host for REST API server')
    parser.add_argument('--no-api', action='store_true', help='Disable REST API server')
    
    args = parser.parse_args()
    
    # 创建服务器实例
    server_instance = ObjectSearchServer(prompt_name=args.prompt)
    
    # 启动服务器
    if not server_instance.start():
        logger.error("Failed to start server")
        exit(1)
    
    # 启动API服务器（可选）
    if not args.no_api:
        api_thread = threading.Thread(
            target=run_api_server,
            kwargs={
                'host': args.api_host,
                'port': args.api_port
            }
        )
        api_thread.daemon = True
        api_thread.start()
        logger.info(f"REST API available at http://{args.api_host}:{args.api_port}")
    
    try:
        # 等待主线程结束（如果有API服务器，它会在后台运行）
        if args.no_api:
            # 如果没有API，等待Ctrl+C
            while server_instance.running:
                time.sleep(1)
        else:
            # 如果有API，主线程可以等待API线程
            api_thread.join()
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down...")
    finally:
        # 停止服务器
        server_instance.stop()
    
    logger.info("Server shutdown complete")