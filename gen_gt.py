# generate_tracked_pseudo_gt.py
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm

class SimpleIOUTracker:
    """簡單的 IoU 追蹤器用於生成偽 GT"""
    
    def __init__(self, iou_threshold=0.3, max_age=30):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks = {}
        self.next_id = 1
    
    def compute_iou(self, box1, box2):
        """計算 IoU"""
        x1, y1, w1, h1 = box1[:4]
        x2, y2, w2, h2 = box2[:4]
        
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = w1 * h1 + w2 * h2 - inter
        
        return inter / (union + 1e-6)
    
    def update(self, detections):
        """更新追蹤 (detections: [[x, y, w, h, conf], ...])"""
        matched = {}
        used_dets = set()
        
        # 匹配現有軌跡
        for track_id, track_info in self.tracks.items():
            best_iou = 0
            best_det_idx = -1
            
            for det_idx, det in enumerate(detections):
                if det_idx in used_dets:
                    continue
                
                iou = self.compute_iou(track_info['box'], det[:4])
                
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_det_idx = det_idx
            
            if best_det_idx >= 0:
                matched[track_id] = best_det_idx
                used_dets.add(best_det_idx)
        
        # 更新軌跡
        new_tracks = {}
        for track_id, det_idx in matched.items():
            det = detections[det_idx]
            new_tracks[track_id] = {
                'box': det[:4],
                'conf': det[4],
                'age': 0
            }
        
        # 創建新軌跡
        for det_idx, det in enumerate(detections):
            if det_idx not in used_dets:
                new_tracks[self.next_id] = {
                    'box': det[:4],
                    'conf': det[4],
                    'age': 0
                }
                self.next_id += 1
        
        # 保留未匹配但未超齡的軌跡
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


def generate_tracked_pseudo_gt(
    yolo_weights,
    seq_dir,
    output_dir,
    conf_threshold=0.5,
    iou_threshold=0.3
):
    """生成帶追蹤 ID 的偽 GT"""
    
    seq_name = seq_dir.name
    img_dir = seq_dir / "img1"
    
    if not img_dir.exists():
        print(f"[Skip] {seq_name}: No images")
        return
    
    images = sorted(img_dir.glob("*.jpg"))
    
    if len(images) == 0:
        print(f"[Skip] {seq_name}: No images")
        return
    
    # 載入 YOLO
    yolo_model = YOLO(yolo_weights)
    
    # 初始化追蹤器
    tracker = SimpleIOUTracker(iou_threshold=iou_threshold)
    
    # 建立輸出目錄
    out_seq_dir = output_dir / seq_name / "gt"
    out_seq_dir.mkdir(parents=True, exist_ok=True)
    
    all_tracks = []
    
    print(f"[Processing] {seq_name} - {len(images)} frames")
    
    for img_path in tqdm(images, desc=seq_name, leave=False):
        frame_id = int(img_path.stem)
        frame = cv2.imread(str(img_path))
        
        # YOLO 檢測
        results = yolo_model(frame, conf=conf_threshold, verbose=False)
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confs):
                x_center, y_center, w, h = box
                
                # 轉換為左上角座標
                x_left = x_center - w / 2
                y_top = y_center - h / 2
                
                detections.append([x_left, y_top, w, h, conf])
        
        # 追蹤更新
        tracked = tracker.update(detections)
        
        # 儲存
        for track in tracked:
            track_id, x, y, w, h, conf = track
            
            all_tracks.append([
                frame_id,
                track_id,
                x, y, w, h,
                conf,
                -1, -1
            ])
    
    # 寫入 gt.txt
    gt_file = out_seq_dir / "gt.txt"
    
    with open(gt_file, 'w') as f:
        for track in all_tracks:
            f.write(','.join(map(str, track)) + '\n')
    
    print(f"  [Saved] {len(all_tracks)} tracks -> {gt_file}")
    print(f"  [Stats] Unique IDs: {len(set(t[1] for t in all_tracks))}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo_weights', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--conf_threshold', type=float, default=0.5)
    parser.add_argument('--iou_threshold', type=float, default=0.3)
    
    args = parser.parse_args()
    
    print("="*80)
    print("Generating Tracked Pseudo GT")
    print("="*80)
    print(f"YOLO: {args.yolo_weights}")
    print(f"Data: {args.data_root}")
    print(f"Output: {args.output_dir}")
    print("="*80 + "\n")
    
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    
    sequences = sorted([d for d in data_root.iterdir() if d.is_dir()])
    
    print(f"Found {len(sequences)} sequences\n")
    
    for seq_dir in sequences:
        generate_tracked_pseudo_gt(
            args.yolo_weights,
            seq_dir,
            output_dir,
            args.conf_threshold,
            args.iou_threshold
        )
    
    print("\n✓ Completed!")


if __name__ == "__main__":
    main()