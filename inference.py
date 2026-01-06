# -*- coding: utf-8 -*-
"""
Inference Script for D2MP-DDIM (CORRECTED VERSION)
按照 DiffMOT 論文正確實現：
1. D2MP 預測用於輔助匹配
2. 保存檢測位置，不是預測位置
3. 減少 ID 碎片化
"""
import torch
import numpy as np
import cv2
import yaml
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import warnings

from model import D2MP_DDIM
from ultralytics import YOLO
from enhanced_tracker import EnhancedTracker

warnings.filterwarnings('ignore')


# ==================== Detector & Tracker ====================

class YOLODetector:
    """YOLO 檢測器包裝"""
    def __init__(self, weights_path, conf=0.5):
        self.model = YOLO(weights_path)
        self.conf = conf
        print(f"[YOLO] Loaded: {weights_path}")
    
    def detect(self, frame):
        """檢測一幀 - 返回中心座標格式"""
        results = self.model(frame, conf=self.conf, verbose=False)
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confs):
                detections.append([*box, conf])
        
        return detections


class SimpleTracker:
    """
    IoU Tracker - 修正版
    按照 DiffMOT 論文實現：使用 D2MP 預測輔助匹配
    """
    def __init__(self, iou_threshold=0.3, max_age=30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 1
    
    def compute_iou(self, box1, box2):
        """計算 IoU - 中心座標格式"""
        x1, y1, w1, h1 = box1[:4]
        x2, y2, w2, h2 = box2[:4]
        
        x1_tl = x1 - w1 / 2
        y1_tl = y1 - h1 / 2
        x2_tl = x2 - w2 / 2
        y2_tl = y2 - h2 / 2
        
        xi1 = max(x1_tl, x2_tl)
        yi1 = max(y1_tl, y2_tl)
        xi2 = min(x1_tl + w1, x2_tl + w2)
        yi2 = min(y1_tl + h1, y2_tl + h2)
        
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter
        
        return inter / (union + 1e-6)
    
    def update_with_prediction(self, detections, predicted_positions=None):
        """
        使用預測位置輔助匹配（DiffMOT 方法）
        
        Args:
            detections: YOLO 檢測結果 [[x, y, w, h, conf], ...]
            predicted_positions: {track_id: [x, y, w, h]} 預測位置字典
        
        Returns:
            tracks: [[track_id, x, y, w, h, conf], ...]
        """
        matched = {}
        used_dets = set()
        
        # 第一階段：使用預測位置匹配檢測
        for track_id, track_info in self.tracks.items():
            best_iou = 0
            best_det_idx = -1
            
            # 使用預測位置（如果有）來計算 IoU
            if predicted_positions and track_id in predicted_positions:
                reference_box = predicted_positions[track_id]
            else:
                reference_box = track_info['box']
            
            for det_idx, det in enumerate(detections):
                if det_idx in used_dets:
                    continue
                
                iou = self.compute_iou(reference_box, det[:4])
                
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_det_idx >= 0:
                matched[track_id] = best_det_idx
                used_dets.add(best_det_idx)
        
        # 第二階段：更新軌跡（使用檢測位置，不是預測位置）
        new_tracks = {}
        for track_id, det_idx in matched.items():
            det = detections[det_idx]
            new_tracks[track_id] = {
                'box': det[:4],  # 使用檢測位置
                'conf': det[4],
                'age': 0,
                'boxes': self.tracks[track_id]['boxes'] + [det[:4]]
            }
        
        # 第三階段：創建新軌跡
        for det_idx, det in enumerate(detections):
            if det_idx not in used_dets:
                new_tracks[self.next_id] = {
                    'box': det[:4],
                    'conf': det[4],
                    'age': 0,
                    'boxes': [det[:4]]
                }
                self.next_id += 1
        
        # 第四階段：保留未匹配但未超齡的軌跡
        for track_id, track_info in self.tracks.items():
            if track_id not in matched:
                track_info['age'] += 1
                if track_info['age'] < self.max_age:
                    new_tracks[track_id] = track_info
        
        self.tracks = new_tracks
        
        # 返回當前軌跡
        tracked = []
        for track_id, track_info in self.tracks.items():
            tracked.append([track_id, *track_info['box'], track_info['conf']])
        
        return tracked
    
    def update(self, detections):
        """向後兼容：不使用預測"""
        return self.update_with_prediction(detections, None)
    
    def get_track_history(self, track_id, n_frames):
        """獲取軌跡歷史"""
        if track_id not in self.tracks:
            return None
        
        boxes = self.tracks[track_id]['boxes']
        
        if len(boxes) < n_frames:
            return None
        
        return np.array(boxes[-n_frames:], dtype=np.float32)


# ==================== Model & Prediction ====================

def load_model(config_path, checkpoint_path, device='cuda'):
    """載入模型"""
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model = D2MP_DDIM(config['model']).to(device)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 處理不同的 checkpoint 格式
    if isinstance(checkpoint, dict):
        # 檢查可能的 keys
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'ddpm' in checkpoint:
            state_dict = checkpoint['ddpm']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # 嘗試載入
    try:
        model.load_state_dict(state_dict, strict=True)
        print("✓ Model loaded successfully")
    except RuntimeError as e:
        print(f"Warning: {e}")
        print("Trying with strict=False...")
        model.load_state_dict(state_dict, strict=False)
        print("⚠ Model loaded with strict=False")
    
    model.eval()
    return model, config


def predict_motion(model, history_boxes, device, sampler='ddim'):
    """
    預測運動（用於輔助匹配）
    
    Args:
        model: D2MP 模型
        history_boxes: 歷史框位置 [n_frames, 4]
        device: 設備
        sampler: 採樣器
    
    Returns:
        predicted_box: 預測的下一幀位置 [x, y, w, h]
    """
    try:
        if np.isnan(history_boxes).any() or np.isinf(history_boxes).any():
            return None
        
        history_boxes = np.clip(history_boxes.astype(np.float32), 0, None)
        deltas = np.diff(history_boxes, axis=0)
        
        if np.isnan(deltas).any() or np.isinf(deltas).any():
            return None
        
        condition = np.concatenate([history_boxes[1:], deltas], axis=1)
        
        if np.isnan(condition).any() or np.isinf(condition).any():
            return None
        
        condition_tensor = torch.from_numpy(condition).float().unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred_motion = model.sample(condition_tensor, device=device, sampler=sampler)
        
        pred_motion_np = pred_motion[0].cpu().numpy()
        
        if np.isnan(pred_motion_np).any() or np.isinf(pred_motion_np).any():
            return None
        
        last_box = history_boxes[-1]
        predicted_box = last_box + pred_motion_np
        
        if np.isnan(predicted_box).any() or np.isinf(predicted_box).any():
            return None
        
        predicted_box = np.clip(predicted_box, 0, None)
        
        return predicted_box
    
    except Exception as e:
        return None



# ==================== Visualization (Enhanced) ====================

def get_color_for_id(track_id):
    """為每個 track_id 生成固定且鮮明的顏色"""
    np.random.seed(track_id * 123)
    h = (track_id * 37) % 180
    s = 200 + (track_id * 13) % 56
    v = 200 + (track_id * 17) % 56
    
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    color = tuple(map(int, bgr[0][0]))
    
    return color


def draw_bbox(frame, box, color, label, thickness=3, font_scale=1.0):
    """繪製 bounding box - 增強版（更大更清晰）"""
    x_center, y_center, w, h = box[:4]
    
    x = int(x_center - w / 2)
    y = int(y_center - h / 2)
    w = int(w)
    h = int(h)
    
    x = max(0, x)
    y = max(0, y)
    
    # 繪製加粗的邊界框
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    # 在四個角落繪製加強標記
    corner_len = 15
    corner_thickness = thickness + 1
    cv2.line(frame, (x, y), (x + corner_len, y), color, corner_thickness)
    cv2.line(frame, (x, y), (x, y + corner_len), color, corner_thickness)
    cv2.line(frame, (x + w, y), (x + w - corner_len, y), color, corner_thickness)
    cv2.line(frame, (x + w, y), (x + w, y + corner_len), color, corner_thickness)
    cv2.line(frame, (x, y + h), (x + corner_len, y + h), color, corner_thickness)
    cv2.line(frame, (x, y + h), (x, y + h - corner_len), color, corner_thickness)
    cv2.line(frame, (x + w, y + h), (x + w - corner_len, y + h), color, corner_thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_len), color, corner_thickness)
    
    # 繪製標籤
    if label:
        label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, font_scale, 2)
        label_y = max(y - label_size[1] - 10, label_size[1] + 10)
        
        # 繪製半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, 
                     (x, label_y - label_size[1] - 8), 
                     (x + label_size[0] + 12, label_y + 4), 
                     color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        text_x = x + 6
        text_y = label_y - 4
        cv2.putText(frame, label, (text_x, text_y), 
                   cv2.FONT_HERSHEY_DUPLEX, font_scale, (0, 0, 0), 4)
        cv2.putText(frame, label, (text_x, text_y), 
                   cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 2)


def draw_trajectory(frame, boxes, color, thickness=3, max_points=30):
    """繪製軌跡 - 增強版（帶漸變和圓點）"""
    if len(boxes) < 2:
        return
    
    boxes_to_draw = boxes[-max_points:] if len(boxes) > max_points else boxes
    points = [(int(box[0]), int(box[1])) for box in boxes_to_draw]
    
    for i in range(len(points) - 1):
        alpha = 0.3 + 0.7 * (i + 1) / len(points)
        overlay = frame.copy()
        cv2.line(overlay, points[i], points[i+1], color, thickness, cv2.LINE_AA)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        circle_radius = max(2, thickness // 2)
        cv2.circle(frame, points[i], circle_radius, color, -1)
    
    if len(points) > 0:
        cv2.circle(frame, points[-1], thickness + 2, color, -1)
        cv2.circle(frame, points[-1], thickness + 4, (255, 255, 255), 2)


def draw_prediction_arrow(frame, current_box, predicted_box, color, thickness=3):
    """繪製預測箭頭"""
    x1, y1 = int(current_box[0]), int(current_box[1])
    x2, y2 = int(predicted_box[0]), int(predicted_box[1])
    
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if length < 5:
        return
    
    overlay = frame.copy()
    cv2.arrowedLine(overlay, (x1, y1), (x2, y2), color, thickness, 
                    cv2.LINE_AA, tipLength=0.3)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    cv2.circle(frame, (x2, y2), 8, color, 2)
    cv2.circle(frame, (x2, y2), 3, color, -1)

# ==================== Demo Mode ====================

def infer_demo_video(
    model,
    detector,
    video_path,
    output_video,
    n_frames,
    device,
    sampler='ddim',
    save_images=False,
    image_dir=None
):
    """Demo 模式: YOLO 檢測 + D2MP 預測"""
    print("\n" + "="*60)
    print("Demo Mode")
    print("="*60)
    print(f"Video: {video_path}")
    print(f"Output: {output_video}")
    print("="*60 + "\n")
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if save_images:
        image_dir = Path(image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)
    
    tracker = EnhancedTracker(iou_threshold=0.3, max_age=50, min_hits=3)
    frame_count = 0
    failed_count = 0
    
    print(f"Processing {total_frames} frames...")
    
    with tqdm(total=total_frames) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # YOLO 檢測
            detections = detector.detect(frame)
            
            # D2MP 預測（輔助匹配）
            predicted_positions = {}
            for track_id in list(tracker.tracks.keys()):
                history = tracker.get_track_history(track_id, n_frames)
                if history is not None:
                    pred_box = predict_motion(model, history, device, sampler)
                    if pred_box is not None:
                        predicted_positions[track_id] = pred_box
            
            # 追蹤（使用預測輔助匹配）
            tracks = tracker.update_with_prediction(detections, predicted_positions)
            
            # 可視化（增強版）
            vis_frame = frame.copy()
            
            # 先繪製所有軌跡（在背景層）
            for track in tracks:
                track_id = int(track[0])
                color = get_color_for_id(track_id)
                
                if track_id in tracker.tracks:
                    history_boxes = tracker.tracks[track_id]['boxes']
                    if len(history_boxes) > 1:
                        draw_trajectory(vis_frame, history_boxes, color, thickness=3, max_points=30)
            
            # 繪製當前追蹤框和預測箭頭
            for track in tracks:
                track_id, x, y, w, h, conf = track
                track_id = int(track_id)
                color = get_color_for_id(track_id)
                current_box = [x, y, w, h]
                
                # 繪製當前檢測框
                draw_bbox(vis_frame, current_box, color, f"ID:{track_id}", thickness=3, font_scale=0.9)
                
                # 如果有預測，繪製預測箭頭
                if track_id in predicted_positions:
                    pred_box = predicted_positions[track_id]
                    draw_prediction_arrow(vis_frame, current_box, pred_box, color, thickness=2)
                    
                    # 在預測位置繪製虛線框（更細、更透明）
                    overlay = vis_frame.copy()
                    x_pred = int(pred_box[0] - pred_box[2] / 2)
                    y_pred = int(pred_box[1] - pred_box[3] / 2)
                    w_pred = int(pred_box[2])
                    h_pred = int(pred_box[3])
                    cv2.rectangle(overlay, (x_pred, y_pred), (x_pred + w_pred, y_pred + h_pred), 
                                color, 2)
                    cv2.addWeighted(overlay, 0.4, vis_frame, 0.6, 0, vis_frame)
            
            # 顯示信息（增強版）
            overlay = vis_frame.copy()
            cv2.rectangle(overlay, (5, 5), (450, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, vis_frame, 0.5, 0, vis_frame)
            
            info_lines = [
                f"Frame: {frame_count}/{total_frames}",
                f"Active Tracks: {len(tracks)}",
                f"Predictions: {len(predicted_positions)}"
            ]
            
            for i, line in enumerate(info_lines):
                y_pos = 30 + i * 22
                cv2.putText(vis_frame, line, (15, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            out.write(vis_frame)
            
            if save_images and frame_count % 10 == 0:
                cv2.imwrite(str(image_dir / f"{frame_count:06d}.jpg"), vis_frame)
            
            pbar.update(1)
    
    cap.release()
    out.release()
    
    print(f"\n✓ Saved: {output_video}")
    print(f"Total frames: {frame_count}")
    print(f"Failed predictions: {failed_count}")


# ==================== Test Mode (Corrected) ====================

def infer_sequence_test(
    yolo_detector,
    model,
    seq_dir,
    output_dir,
    n_frames,
    device,
    sampler='ddim',
    save_video=False,
    save_images=False
):
    """
    Test 模式 - 修正版（遵循 DiffMOT 論文）
    
    關鍵修正：
    1. D2MP 預測用於輔助匹配
    2. 保存檢測位置，不是預測位置
    3. 減少 ID 碎片化
    """
    seq_name = seq_dir.name
    img_dir = seq_dir / "img1"
    
    if not img_dir.exists():
        print(f"[Skip] {seq_name}: images not found")
        return
    
    out_seq_dir = Path(output_dir) / seq_name
    out_seq_dir.mkdir(parents=True, exist_ok=True)
    
    images = sorted(img_dir.glob("*.jpg"))
    
    if len(images) == 0:
        print(f"[Skip] {seq_name}: No images")
        return
    
    first_frame = cv2.imread(str(images[0]))
    height, width = first_frame.shape[:2]
    
    video_writer = None
    if save_video:
        video_path = out_seq_dir / f"{seq_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, (width, height))
    
    vis_dir = None
    if save_images:
        vis_dir = out_seq_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
    
    # 初始化追蹤器
    tracker = SimpleTracker(iou_threshold=0.35, max_age=50)
    predictions = []
    
    print(f"\n[{seq_name}] Processing {len(images)} frames...")
    
    for img_path in tqdm(images, desc=seq_name, leave=False):
        frame_id = int(img_path.stem)
        frame = cv2.imread(str(img_path))
        
        # ===== 第一步：YOLO 檢測 =====
        detections = yolo_detector.detect(frame)
        
        # ===== 第二步：D2MP 預測（輔助匹配用）=====
        predicted_positions = {}
        for track_id in list(tracker.tracks.keys()):
            history = tracker.get_track_history(track_id, n_frames)
            
            if history is not None:
                pred_box = predict_motion(model, history, device, sampler)
                
                if pred_box is not None:
                    predicted_positions[track_id] = pred_box
        
        # ===== 第三步：使用預測輔助匹配 =====
        tracks = tracker.update_with_prediction(detections, predicted_positions)
        
        # ===== 第四步：保存追蹤結果（檢測位置，不是預測位置）=====
        for track in tracks:
            track_id, x_center, y_center, w, h, conf = track
            
            # 轉換為 MOT 格式（左上角）
            x_left = x_center - w / 2
            y_top = y_center - h / 2
            
            # 保存檢測位置（不是預測位置）
            predictions.append([
                frame_id,      # 當前幀
                track_id,
                x_left, y_top, w, h,
                conf, -1, -1
            ])
        
        # ===== 可視化 =====
        if save_video or save_images:
            vis_frame = frame.copy()
            
            # 繪製當前追蹤
            for track in tracks:
                track_id, x, y, w, h, conf = track
                draw_bbox(vis_frame, [x, y, w, h], (0, 255, 0), f"ID:{track_id}")
            
            # 繪製預測框（用於可視化D2MP效果）
            for track_id, pred_box in predicted_positions.items():
                draw_bbox(vis_frame, pred_box, (0, 0, 255), f"Pred:{track_id}", 1)
            
            info = f"Frame: {frame_id} | Tracks: {len(tracks)} | Predictions: {len(predicted_positions)}"
            cv2.putText(vis_frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if video_writer:
                video_writer.write(vis_frame)
            
            if save_images and frame_id % 10 == 0:
                cv2.imwrite(str(vis_dir / f"{frame_id:08d}.jpg"), vis_frame)
    
    # 保存預測
    pred_file = out_seq_dir / "pred.txt"
    with open(pred_file, 'w') as f:
        for pred in predictions:
            f.write(','.join(map(str, pred)) + '\n')
    
    print(f"  [Save] {len(predictions)} predictions")
    
    if video_writer:
        video_writer.release()
        print(f"  [Save] {video_path}")


def infer_test_mode(
    yolo_weights,
    model,
    data_root,
    output_dir,
    n_frames,
    device,
    conf_threshold=0.5,
    sampler='ddim',
    save_video=False,
    save_images=False
):
    """Test 模式：使用 YOLO 檢測 + D2MP 預測（修正版）"""
    print("\n" + "="*60)
    print("Test Mode (CORRECTED - Following DiffMOT Paper)")
    print("="*60)
    print(f"YOLO weights: {yolo_weights}")
    print(f"Data root: {data_root}")
    print(f"Output: {output_dir}")
    print(f"History frames: {n_frames}")
    print("="*60)
    print("Key improvements:")
    print("  1. D2MP prediction for matching assistance")
    print("  2. Save detection positions (not predictions)")
    print("  3. Reduce ID fragmentation")
    print("="*60 + "\n")
    
    # 載入 YOLO
    yolo_detector = YOLODetector(yolo_weights, conf=conf_threshold)
    
    data_root = Path(data_root)
    sequences = sorted([d for d in data_root.iterdir() if d.is_dir()])
    
    print(f"Found {len(sequences)} sequences\n")
    
    for seq_dir in sequences:
        infer_sequence_test(
            yolo_detector, model, seq_dir, output_dir, n_frames, device, sampler,
            save_video, save_images
        )


# ==================== Val Mode (With GT) ====================

def infer_sequence_val(
    model,
    seq_dir,
    output_dir,
    n_frames,
    device,
    sampler='ddim',
    save_video=False,
    save_images=False
):
    """Val 模式：使用 GT 進行預測"""
    seq_name = seq_dir.name
    gt_file = seq_dir / "gt" / "gt.txt"
    img_dir = seq_dir / "img1"
    
    if not gt_file.exists() or not img_dir.exists():
        print(f"[Skip] {seq_name}: GT or images not found")
        return
    
    out_seq_dir = Path(output_dir) / seq_name
    out_seq_dir.mkdir(parents=True, exist_ok=True)
    
    import pandas as pd
    df = pd.read_csv(
        gt_file, header=None,
        names=['frame_id', 'track_id', 'x', 'y', 'w', 'h', 'conf', 'cls', 'vis']
    )
    
    gt_by_frame = {}
    for _, row in df.iterrows():
        frame = int(row['frame_id'])
        if frame not in gt_by_frame:
            gt_by_frame[frame] = []
        
        x_left = row['x']
        y_top = row['y']
        w = row['w']
        h = row['h']
        
        x_center = x_left + w / 2
        y_center = y_top + h / 2
        
        gt_by_frame[frame].append({
            'track_id': int(row['track_id']),
            'box': [x_center, y_center, w, h]
        })
    
    images = sorted(img_dir.glob("*.jpg"))
    
    if len(images) == 0:
        print(f"[Skip] {seq_name}: No images")
        return
    
    first_frame = cv2.imread(str(images[0]))
    height, width = first_frame.shape[:2]
    
    video_writer = None
    if save_video:
        video_path = out_seq_dir / f"{seq_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 30, (width, height))
    
    if save_images:
        vis_dir = out_seq_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
    
    track_histories = {}
    predictions = []
    
    print(f"\n[{seq_name}] Processing {len(images)} frames...")
    
    for img_path in tqdm(images, desc=seq_name, leave=False):
        frame_id = int(img_path.stem)
        frame = cv2.imread(str(img_path))
        
        if frame_id not in gt_by_frame:
            continue
        
        current_gt = gt_by_frame[frame_id]
        
        vis_frame = frame.copy() if (save_video or save_images) else None
        
        for gt_obj in current_gt:
            track_id = gt_obj['track_id']
            box = gt_obj['box']
            
            if track_id not in track_histories:
                track_histories[track_id] = []
            
            track_histories[track_id].append(box)
            
            if len(track_histories[track_id]) >= n_frames:
                history = np.array(track_histories[track_id][-n_frames:], dtype=np.float32)
                
                pred_box = predict_motion(model, history, device, sampler)
                
                if pred_box is not None:
                    x_center, y_center, w, h = pred_box
                    x_left = x_center - w / 2
                    y_top = y_center - h / 2
                    
                    predictions.append([
                        frame_id + 1,
                        track_id,
                        x_left, y_top, w, h,
                        1.0, -1, -1
                    ])
                    
                    if vis_frame is not None:
                        draw_bbox(vis_frame, pred_box, (0, 0, 255), f"Pred:{track_id}")
            
            if vis_frame is not None:
                draw_bbox(vis_frame, box, (0, 255, 0), f"GT:{track_id}")
        
        if vis_frame is not None:
            info = f"Frame: {frame_id} | Objects: {len(current_gt)}"
            cv2.putText(vis_frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if video_writer:
            video_writer.write(vis_frame)
        
        if save_images and frame_id % 10 == 0:
            cv2.imwrite(str(vis_dir / f"{frame_id:08d}.jpg"), vis_frame)
    
    pred_file = out_seq_dir / "pred.txt"
    with open(pred_file, 'w') as f:
        for pred in predictions:
            f.write(','.join(map(str, pred)) + '\n')
    
    print(f"  [Save] {len(predictions)} predictions")
    
    if video_writer:
        video_writer.release()


def infer_val_mode(
    model,
    data_root,
    output_dir,
    n_frames,
    device,
    sampler='ddim',
    save_video=False,
    save_images=False
):
    """Val 模式：使用 GT 進行預測"""
    print("\n" + "="*60)
    print("Val Mode (With Ground Truth)")
    print("="*60)
    print(f"Data root: {data_root}")
    print(f"Output: {output_dir}")
    print("="*60 + "\n")
    
    data_root = Path(data_root)
    sequences = sorted([d for d in data_root.iterdir() if d.is_dir()])
    
    print(f"Found {len(sequences)} sequences\n")
    
    for seq_dir in sequences:
        infer_sequence_val(
            model, seq_dir, output_dir, n_frames, device, sampler,
            save_video, save_images
        )


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description='D2MP-DDIM Inference (CORRECTED)')
    
    parser.add_argument('--mode', type=str, required=True,
                        choices=['demo', 'test', 'val'],
                        help='Inference mode')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--n_frames', type=int, default=5,
                        help='Number of history frames for D2MP')
    parser.add_argument('--sampler', type=str, default='ddim',
                        choices=['ddim', 'ddpm'],
                        help='Sampling method')
    
    # Demo mode
    parser.add_argument('--video', type=str,
                        help='Input video path (demo mode)')
    parser.add_argument('--output_video', type=str,
                        help='Output video path (demo mode)')
    parser.add_argument('--yolo_weights', type=str,
                        help='YOLO weights path (demo/test mode)')
    
    # Test/Val mode
    parser.add_argument('--data_root', type=str,
                        help='Data root directory (test/val mode)')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory (test/val mode)')
    parser.add_argument('--conf_threshold', type=float, default=0.5,
                        help='YOLO confidence threshold')
    
    # Visualization
    parser.add_argument('--save_video', action='store_true',
                        help='Save visualization videos')
    parser.add_argument('--save_images', action='store_true',
                        help='Save visualization images')
    
    args = parser.parse_args()
    
    # 載入模型
    print("="*60)
    print("D2MP-DDIM Inference (CORRECTED VERSION)")
    print("Following DiffMOT Paper Methodology")
    print("="*60)
    
    model, config = load_model(args.config, args.checkpoint, args.device)
    
    if args.mode == 'demo':
        if not args.video or not args.output_video or not args.yolo_weights:
            raise ValueError("Demo mode requires --video, --output_video, --yolo_weights")
        
        detector = YOLODetector(args.yolo_weights, conf=args.conf_threshold)
        
        infer_demo_video(
            model, detector, args.video, args.output_video,
            args.n_frames, args.device, args.sampler,
            args.save_images, args.output_dir
        )
    
    elif args.mode == 'test':
        if not args.data_root or not args.output_dir or not args.yolo_weights:
            raise ValueError("Test mode requires --data_root, --output_dir, --yolo_weights")
        
        infer_test_mode(
            args.yolo_weights, model, args.data_root, args.output_dir,
            args.n_frames, args.device, args.conf_threshold, args.sampler,
            args.save_video, args.save_images
        )
    
    elif args.mode == 'val':
        if not args.data_root or not args.output_dir:
            raise ValueError("Val mode requires --data_root, --output_dir")
        
        infer_val_mode(
            model, args.data_root, args.output_dir,
            args.n_frames, args.device, args.sampler,
            args.save_video, args.save_images
        )
    
    print("\n" + "="*60)
    print("✓ Inference completed!")
    print("="*60)


if __name__ == "__main__":
    main()