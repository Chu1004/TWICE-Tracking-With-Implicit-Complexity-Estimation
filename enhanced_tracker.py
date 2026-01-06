#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Tracker - 完全兼容 SimpleTracker 接口
同時提供 Hungarian Matching + Kalman Filter 的強大功能
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker:
    """使用 Kalman Filter 追蹤單個目標"""
    
    count = 0
    
    def __init__(self, bbox):
        """
        初始化 Kalman Filter
        狀態: [x, y, w, h, vx, vy, vw, vh]
        測量: [x, y, w, h]
        """
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # 狀態轉移矩陣
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ])
        
        # 測量矩陣
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ])
        
        # 測量噪聲
        self.kf.R *= 10.
        
        # 過程噪聲
        self.kf.P[4:, 4:] *= 1000.
        self.kf.P *= 10.
        
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01
        
        # 初始化狀態
        self.kf.x[:4] = np.array(bbox).reshape(4, 1)
        
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
    
    def update(self, bbox):
        """更新 Kalman Filter"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(np.array(bbox).reshape(4, 1))
    
    def predict(self):
        """預測下一個狀態"""
        if self.kf.x[2] + self.kf.x[6] <= 0:
            self.kf.x[6] = 0
        if self.kf.x[3] + self.kf.x[7] <= 0:
            self.kf.x[7] = 0
        
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(self.kf.x[:4].flatten())
        
        return self.kf.x[:4].flatten()
    
    def get_state(self):
        """獲取當前狀態"""
        return self.kf.x[:4].flatten()


class EnhancedTracker:
    """
    Enhanced Tracker - 完全兼容 SimpleTracker 接口
    
    特性：
    1. Hungarian Matching (全局最優)
    2. Kalman Filter (平滑運動)
    3. 兼容 SimpleTracker 的所有接口
    """
    
    def __init__(self, iou_threshold=0.3, max_age=50, min_hits=3):
        """
        初始化追蹤器
        
        Args:
            iou_threshold: IoU 匹配閾值
            max_age: 追蹤丟失後保留的最大幀數
            min_hits: 輸出追蹤前需要的最小匹配次數
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        
        self.trackers = []
        self.frame_count = 0
        self.next_id = 1
        
        # 兼容 SimpleTracker 接口
        self.tracks = {}  # {track_id: {'box': [x,y,w,h], 'history': [[x,y,w,h], ...]}}
        
        print(f"[EnhancedTracker] 初始化")
        print(f"  iou_threshold: {iou_threshold}")
        print(f"  max_age: {max_age}")
        print(f"  min_hits: {min_hits}")
    
    def compute_iou(self, box1, box2):
        """計算 IoU"""
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
    
    def associate_detections_to_trackers(self, detections, trackers_pred, d2mp_predictions=None):
        """
        使用 Hungarian 算法進行匹配
        
        Args:
            detections: numpy array of [x, y, w, h, conf]
            trackers_pred: numpy array of [x, y, w, h] (Kalman predictions)
            d2mp_predictions: dict {track_id: [x, y, w, h]} (optional)
        
        Returns:
            matches, unmatched_dets, unmatched_trks
        """
        if len(trackers_pred) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0,), dtype=int)
        
        if len(detections) == 0:
            return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(trackers_pred))
        
        # 構建 cost matrix
        iou_matrix = np.zeros((len(detections), len(trackers_pred)), dtype=np.float32)
        
        for d, det in enumerate(detections):
            for t, trk_pred in enumerate(trackers_pred):
                # 優先使用 D2MP 預測
                if d2mp_predictions is not None:
                    # 找到對應的 track_id
                    trk = self.trackers[t]
                    track_id = trk.id + 1
                    if track_id in d2mp_predictions:
                        pred_box = d2mp_predictions[track_id]
                        iou_matrix[d, t] = self.compute_iou(det, pred_box)
                        continue
                
                # 使用 Kalman 預測
                iou_matrix[d, t] = self.compute_iou(det, trk_pred)
        
        # Hungarian 匹配
        cost_matrix = 1 - iou_matrix
        matched_indices = linear_sum_assignment(cost_matrix)
        matched_indices = np.array(list(zip(*matched_indices)))
        
        # 過濾低 IoU 的匹配
        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                continue
            matches.append(m.reshape(1, 2))
        
        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        # 找出未匹配的檢測和追蹤
        unmatched_detections = []
        for d in range(len(detections)):
            if len(matches) == 0 or d not in matches[:, 0]:
                unmatched_detections.append(d)
        
        unmatched_trackers = []
        for t in range(len(trackers_pred)):
            if len(matches) == 0 or t not in matches[:, 1]:
                unmatched_trackers.append(t)
        
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    
    def update_with_prediction(self, detections, predicted_positions=None):
        """
        兼容 SimpleTracker 接口的更新方法
        
        Args:
            detections: list of [x, y, w, h, conf]
            predicted_positions: dict {track_id: [x, y, w, h]} (D2MP predictions)
        
        Returns:
            dict {track_id: [x, y, w, h, conf]}
        """
        self.frame_count += 1
        
        # 轉換為 numpy array
        if isinstance(detections, list):
            if len(detections) == 0:
                detections = np.empty((0, 5))
            else:
                detections = np.array(detections)
        
        # 1. 獲取 Kalman 預測
        trks = []
        to_del = []
        
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks.append(pos)
            
            if np.any(np.isnan(pos)):
                to_del.append(t)
        
        # 刪除無效追蹤器
        for t in reversed(to_del):
            self.trackers.pop(t)
        
        trks = np.array(trks) if len(trks) > 0 else np.empty((0, 4))
        
        # 2. 匹配
        matched, unmatched_dets, unmatched_trks = self.associate_detections_to_trackers(
            detections, trks, predicted_positions
        )
        
        # 3. 更新匹配的追蹤器
        for m in matched:
            det_idx, trk_idx = m
            self.trackers[trk_idx].update(detections[det_idx][:4])
        
        # 4. 為未匹配的檢測創建新追蹤器
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i][:4])
            self.trackers.append(trk)
        
        # 5. 更新 tracks 字典（兼容 SimpleTracker）
        self.tracks = {}
        
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            track_id = trk.id + 1
            
            # 只輸出達到最小 hits 且最近有更新的追蹤
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # 兼容 SimpleTracker 格式
                self.tracks[track_id] = {
                    'box': d,
                    'boxes': trk.history[-30:] if len(trk.history) > 0 else [d],  # 用於繪製軌跡
                    'history': trk.history[-10:] if len(trk.history) > 0 else [d],  # 用於 D2MP 預測
                    'age': trk.age,
                    'hits': trk.hits,
                    'conf': 1.0
                }
            
            i -= 1
            
            # 刪除死亡追蹤器
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        
        # 6. 返回兼容格式 - 轉換為列表
        # 格式: [[track_id, x, y, w, h, conf], ...]
        result = []
        for track_id, track_data in self.tracks.items():
            box = track_data['box']
            # 格式: [track_id, x, y, w, h, conf]
            track_array = [track_id] + list(box) + [1.0]
            result.append(track_array)
        
        return result
    
    def get_track_history(self, track_id, n_frames):
        """
        獲取軌跡歷史（兼容 SimpleTracker）
        
        Args:
            track_id: 軌跡 ID
            n_frames: 需要的歷史幀數
        
        Returns:
            numpy array of [[x,y,w,h], ...] 或 None
        """
        if track_id not in self.tracks:
            return None
        
        history = self.tracks[track_id]['history']
        
        if len(history) < n_frames:
            return None
        
        return np.array(history[-n_frames:], dtype=np.float32)