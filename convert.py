# -*- coding: utf-8 -*-
"""
Convert DanceTrack (MOT format) to YOLO format
將 DanceTrack MOT 格式轉換為 YOLO 格式
"""
import os
import cv2
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm
import argparse


def convert_mot_to_yolo(mot_root, yolo_root, splits=['train', 'val', 'test']):
    """
    轉換 MOT 格式到 YOLO 格式
    
    Args:
        mot_root: MOT 格式根目錄 (dataset/dancetrack)
        yolo_root: YOLO 格式輸出目錄 (dataset/dancetrack_yolo)
        splits: 要轉換的 split
    """
    mot_root = Path(mot_root)
    yolo_root = Path(yolo_root)
    
    print("="*60)
    print("MOT to YOLO Format Converter")
    print("="*60)
    print(f"MOT root: {mot_root}")
    print(f"YOLO root: {yolo_root}")
    print(f"Splits: {splits}")
    print("="*60 + "\n")
    
    # 創建 YOLO 目錄結構
    for split in splits:
        (yolo_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (yolo_root / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # 統計
    total_images = 0
    total_boxes = 0
    
    # 轉換每個 split
    for split in splits:
        split_dir = mot_root / split
        
        if not split_dir.exists():
            print(f"[Skip] {split}: Directory not found")
            continue
        
        sequences = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        
        print(f"\n[{split.upper()}] Converting {len(sequences)} sequences...")
        
        for seq_dir in tqdm(sequences, desc=f"{split}"):
            seq_name = seq_dir.name
            
            # 路徑
            img_dir = seq_dir / 'img1'
            gt_file = seq_dir / 'gt' / 'gt.txt'
            seqinfo_file = seq_dir / 'seqinfo.ini'
            
            if not img_dir.exists() or not gt_file.exists():
                print(f"  [Skip] {seq_name}: Missing img1 or gt.txt")
                continue
            
            # 讀取圖片尺寸
            img_width, img_height = None, None
            
            # 1. 從 seqinfo.ini 讀取
            if seqinfo_file.exists():
                with open(seqinfo_file, 'r') as f:
                    for line in f:
                        if 'imWidth' in line:
                            img_width = int(line.split('=')[1].strip())
                        if 'imHeight' in line:
                            img_height = int(line.split('=')[1].strip())
            
            # 2. 如果沒有 seqinfo，讀取第一張圖片
            if img_width is None or img_height is None:
                first_img = next(img_dir.glob('*.jpg'), None)
                if first_img:
                    img = cv2.imread(str(first_img))
                    img_height, img_width = img.shape[:2]
                else:
                    print(f"  [Skip] {seq_name}: Cannot determine image size")
                    continue
            
            # 讀取 GT
            try:
                df = pd.read_csv(
                    gt_file, header=None,
                    names=['frame_id', 'track_id', 'x', 'y', 'w', 'h', 
                           'conf', 'cls', 'vis']
                )
            except Exception as e:
                print(f"  [Skip] {seq_name}: Failed to read gt.txt - {e}")
                continue
            
            # 按幀分組
            frames = df.groupby('frame_id')
            
            # 處理每一幀
            for frame_id, frame_df in frames:
                # 源圖片路徑
                src_img = img_dir / f"{frame_id:06d}.jpg"
                if not src_img.exists():
                    continue
                
                # 目標圖片路徑: seqname_frameid.jpg
                dst_img_name = f"{seq_name}_{frame_id:06d}.jpg"
                dst_img = yolo_root / 'images' / split / dst_img_name
                dst_label = yolo_root / 'labels' / split / f"{seq_name}_{frame_id:06d}.txt"
                
                # 複製圖片
                shutil.copy2(src_img, dst_img)
                total_images += 1
                
                # 轉換標註
                yolo_labels = []
                
                for _, row in frame_df.iterrows():
                    # MOT 格式: x, y, w, h (左上角坐標 + 寬高)
                    x, y, w, h = row['x'], row['y'], row['w'], row['h']
                    
                    # 過濾無效框
                    if w <= 0 or h <= 0:
                        continue
                    
                    # 轉換為 YOLO 格式: class x_center y_center width height (歸一化)
                    x_center = (x + w / 2) / img_width
                    y_center = (y + h / 2) / img_height
                    w_norm = w / img_width
                    h_norm = h / img_height
                    
                    # 限制在 [0, 1]
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    w_norm = max(0, min(1, w_norm))
                    h_norm = max(0, min(1, h_norm))
                    
                    # YOLO 格式: class_id x_center y_center width height
                    # DanceTrack 只有一個類別 (person = 0)
                    yolo_labels.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                    total_boxes += 1
                
                # 保存標註
                with open(dst_label, 'w') as f:
                    f.write('\n'.join(yolo_labels))
        
        print(f"  ✓ {split} completed")
    
    # 創建 dataset.yaml
    yaml_content = f"""# DanceTrack YOLO Format
path: {yolo_root.absolute()}
train: images/train
val: images/val
test: images/test

# Classes
nc: 1
names: ['person']
"""
    
    yaml_file = yolo_root / 'dataset.yaml'
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print("\n" + "="*60)
    print("Conversion Complete!")
    print("="*60)
    print(f"Total images: {total_images}")
    print(f"Total boxes: {total_boxes}")
    print(f"\nDataset config: {yaml_file}")
    print("\nNext step:")
    print(f"  python train_yolo.py --data_yaml {yaml_file} --model s")
    print("="*60)


def verify_conversion(yolo_root, num_samples=5):
    """
    驗證轉換結果
    
    Args:
        yolo_root: YOLO 格式目錄
        num_samples: 驗證樣本數
    """
    print("\n" + "="*60)
    print("Verifying Conversion")
    print("="*60)
    
    yolo_root = Path(yolo_root)
    
    for split in ['train', 'val']:
        img_dir = yolo_root / 'images' / split
        label_dir = yolo_root / 'labels' / split
        
        if not img_dir.exists():
            print(f"[{split}] Skip - directory not found")
            continue
        
        images = list(img_dir.glob('*.jpg'))
        labels = list(label_dir.glob('*.txt'))
        
        print(f"\n[{split.upper()}]")
        print(f"  Images: {len(images)}")
        print(f"  Labels: {len(labels)}")
        
        # 檢查樣本
        for i, img_path in enumerate(images[:num_samples]):
            label_path = label_dir / (img_path.stem + '.txt')
            
            if not label_path.exists():
                print(f"  ✗ Missing label: {label_path.name}")
                continue
            
            # 讀取標註
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # 讀取圖片
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            
            print(f"  ✓ {img_path.name}")
            print(f"    - Size: {w}x{h}")
            print(f"    - Boxes: {len(lines)}")
            
            # 檢查第一個框
            if len(lines) > 0:
                parts = lines[0].strip().split()
                print(f"    - Sample box: class={parts[0]}, x={parts[1]}, y={parts[2]}, w={parts[3]}, h={parts[4]}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Convert MOT to YOLO format")
    
    parser.add_argument(
        '--mot_root',
        type=str,
        required=True,
        help='MOT format root directory (e.g., dataset/dancetrack)'
    )
    
    parser.add_argument(
        '--yolo_root',
        type=str,
        required=True,
        help='YOLO format output directory (e.g., dataset/dancetrack_yolo)'
    )
    
    parser.add_argument(
        '--splits',
        type=str,
        nargs='+',
        default=['train', 'val', 'test'],
        help='Splits to convert'
    )
    
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify conversion after completion'
    )
    
    args = parser.parse_args()
    
    # 轉換
    convert_mot_to_yolo(args.mot_root, args.yolo_root, args.splits)
    
    # 驗證
    if args.verify:
        verify_conversion(args.yolo_root)


if __name__ == "__main__":
    main()