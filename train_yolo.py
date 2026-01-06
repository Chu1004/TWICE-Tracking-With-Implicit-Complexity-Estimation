# -*- coding: utf-8 -*-
"""
Fine-tune YOLO on Target Domain (Test Video)
在目標域（測試視頻）上 Fine-tune YOLO
"""
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
import shutil


def extract_frames_from_video(video_path, output_dir, num_frames=20, method='uniform'):
    """
    從測試視頻中抽取幀用於 fine-tuning
    
    Args:
        video_path: 測試視頻路徑
        output_dir: 輸出目錄
        num_frames: 抽取的幀數
        method: 'uniform' (均勻抽取) 或 'diverse' (多樣性抽取)
    """
    print("="*60)
    print("Step 1: Extract Frames from Test Video")
    print("="*60 + "\n")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {video_path}")
    print(f"Total frames: {total_frames}")
    print(f"Extracting: {num_frames} frames")
    print(f"Method: {method}\n")
    
    if method == 'uniform':
        # 均勻抽取
        step = total_frames // num_frames
        frame_indices = [i * step for i in range(num_frames)]
    else:
        # 多樣性抽取（開頭、中間、結尾各取一些）
        frame_indices = []
        # 開頭 30%
        frame_indices.extend(np.linspace(0, total_frames * 0.3, num_frames // 3, dtype=int))
        # 中間 40%
        frame_indices.extend(np.linspace(total_frames * 0.3, total_frames * 0.7, num_frames // 3, dtype=int))
        # 結尾 30%
        frame_indices.extend(np.linspace(total_frames * 0.7, total_frames - 1, num_frames - 2*(num_frames//3), dtype=int))
    
    extracted = []
    
    for i, frame_id in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ret, frame = cap.read()
        
        if ret:
            output_path = output_dir / f"frame_{frame_id:06d}.jpg"
            cv2.imwrite(str(output_path), frame)
            extracted.append(output_path)
            print(f"  ✓ Extracted: {output_path.name} (frame {frame_id})")
    
    cap.release()
    
    print(f"\n✓ Extracted {len(extracted)} frames to: {output_dir}")
    
    return extracted


def auto_label_with_yolo(image_paths, yolo_weights, output_label_dir, conf_threshold=0.25):
    """
    使用當前 YOLO 模型自動標註抽取的幀
    
    Args:
        image_paths: 圖片路徑列表
        yolo_weights: YOLO 權重路徑
        output_label_dir: 標註輸出目錄
        conf_threshold: 置信度閾值
    """
    print("\n" + "="*60)
    print("Step 2: Auto-label with Current YOLO Model")
    print("="*60 + "\n")
    
    output_label_dir = Path(output_label_dir)
    output_label_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading YOLO model: {yolo_weights}")
    model = YOLO(yolo_weights)
    
    print(f"Confidence threshold: {conf_threshold}")
    print(f"Output labels to: {output_label_dir}\n")
    
    for img_path in image_paths:
        # 讀取圖片尺寸
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        # YOLO 檢測
        results = model(img, conf=conf_threshold, verbose=False)
        
        # 轉換為 YOLO 格式標註
        label_path = output_label_dir / (img_path.stem + '.txt')
        
        with open(label_path, 'w') as f:
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confs):
                    x_center, y_center, width, height = box
                    
                    # 歸一化座標
                    x_norm = x_center / w
                    y_norm = y_center / h
                    w_norm = width / w
                    h_norm = height / h
                    
                    # YOLO 格式: class x_center y_center width height
                    f.write(f"0 {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                
                print(f"  ✓ {img_path.name}: {len(boxes)} detections → {label_path.name}")
            else:
                print(f"  ⚠ {img_path.name}: No detections (empty label)")
    
    print(f"\n✓ Auto-labeled {len(image_paths)} images")
    print(f"\n⚠️  IMPORTANT: Please manually review and correct these labels!")
    print(f"   Tool recommendation: LabelImg or Roboflow")
    print(f"   Labels location: {output_label_dir}")


def create_finetune_dataset(image_dir, label_dir, output_root, train_ratio=0.8):
    """
    創建 Fine-tune 數據集（YOLO 格式）
    
    Args:
        image_dir: 圖片目錄
        label_dir: 標註目錄
        output_root: 輸出根目錄
        train_ratio: 訓練集比例
    """
    print("\n" + "="*60)
    print("Step 3: Create Fine-tune Dataset")
    print("="*60 + "\n")
    
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    output_root = Path(output_root)
    
    # 創建目錄結構
    (output_root / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_root / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_root / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_root / 'labels' / 'val').mkdir(parents=True, exist_ok=True)
    
    # 收集圖片和標註
    images = sorted(list(image_dir.glob('*.jpg')))
    
    # 分割訓練/驗證集
    import random
    random.shuffle(images)
    
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    print(f"Total images: {len(images)}")
    print(f"Train: {len(train_images)}")
    print(f"Val: {len(val_images)}\n")
    
    # 複製文件
    for split, img_list in [('train', train_images), ('val', val_images)]:
        for img_path in img_list:
            # 複製圖片
            shutil.copy(img_path, output_root / 'images' / split / img_path.name)
            
            # 複製標註
            label_path = label_dir / (img_path.stem + '.txt')
            if label_path.exists():
                shutil.copy(label_path, output_root / 'labels' / split / label_path.name)
    
    # 創建 dataset.yaml
    import yaml
    
    yaml_content = {
        'path': str(output_root.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 1,
        'names': ['person']
    }
    
    yaml_path = output_root / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"✓ Dataset created: {output_root}")
    print(f"✓ Config file: {yaml_path}")
    
    return str(yaml_path)


def finetune_yolo(
    base_weights,
    finetune_data_yaml,
    epochs=50,
    lr0=0.001,  # 較小的學習率用於 fine-tune
    batch=8,
    device=0,
    project='runs/detect',
    name='yolo11_finetuned'
):
    """
    Fine-tune YOLO 模型
    
    Args:
        base_weights: 基礎模型權重
        finetune_data_yaml: Fine-tune 數據集配置
        epochs: Fine-tune epochs（通常較少）
        lr0: 初始學習率（應該比從頭訓練小）
        batch: Batch size
        device: GPU 設備
        project: 輸出項目目錄
        name: 實驗名稱
    """
    print("\n" + "="*60)
    print("Step 4: Fine-tune YOLO Model")
    print("="*60 + "\n")
    
    print(f"Base weights: {base_weights}")
    print(f"Fine-tune data: {finetune_data_yaml}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr0} (lower for fine-tuning)")
    print(f"Batch size: {batch}\n")
    
    # 載入基礎模型
    model = YOLO(base_weights)
    
    # Fine-tune
    results = model.train(
        data=finetune_data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=batch,
        workers=2,
        device=device,
        project=project,
        name=name,
        lr0=lr0,  # 較小的學習率
        lrf=0.01,
        optimizer='AdamW',
        patience=20,
        verbose=True,
        save=True,
        plots=True,
        val=True,
        # Fine-tune 建議減少增強
        augment=True,
        mosaic=0.5,  # 減少 mosaic
        mixup=0.0    # 關閉 mixup
    )
    
    print("\n" + "="*60)
    print("Fine-tuning completed!")
    print("="*60)
    
    # 驗證
    print("\n[Validation]")
    metrics = model.val()
    
    print(f"  mAP50: {metrics.box.map50:.4f}")
    print(f"  mAP50-95: {metrics.box.map:.4f}")
    print(f"  Precision: {metrics.box.mp:.4f}")
    print(f"  Recall: {metrics.box.mr:.4f}")
    
    weights_path = Path(project) / name / 'weights' / 'best.pt'
    
    print(f"\n✓ Fine-tuned weights: {weights_path}")
    
    return str(weights_path)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLO on Target Domain")
    
    # Step 1: 抽取幀
    parser.add_argument('--video', type=str, required=True,
                        help='Test video path')
    parser.add_argument('--num_frames', type=int, default=20,
                        help='Number of frames to extract')
    parser.add_argument('--extract_method', type=str, default='diverse',
                        choices=['uniform', 'diverse'],
                        help='Frame extraction method')
    
    # Step 2: 自動標註
    parser.add_argument('--base_weights', type=str, required=True,
                        help='Base YOLO weights for auto-labeling')
    parser.add_argument('--conf_threshold', type=float, default=0.25,
                        help='Confidence threshold for auto-labeling')
    
    # Step 3: 數據集配置
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Train/val split ratio')
    
    # Step 4: Fine-tune 參數
    parser.add_argument('--finetune_epochs', type=int, default=50,
                        help='Fine-tune epochs')
    parser.add_argument('--finetune_lr', type=float, default=0.001,
                        help='Fine-tune learning rate')
    parser.add_argument('--batch', type=int, default=8,
                        help='Batch size for fine-tuning')
    parser.add_argument('--device', type=int, default=0,
                        help='GPU device')
    
    # 輸出配置
    parser.add_argument('--output_root', type=str, default='finetune_data',
                        help='Output root directory')
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='Training project directory')
    parser.add_argument('--name', type=str, default='yolo11_finetuned',
                        help='Experiment name')
    
    # 控制選項
    parser.add_argument('--skip_extract', action='store_true',
                        help='Skip frame extraction (use existing)')
    parser.add_argument('--skip_label', action='store_true',
                        help='Skip auto-labeling (use existing)')
    parser.add_argument('--skip_finetune', action='store_true',
                        help='Skip fine-tuning (only prepare data)')
    
    args = parser.parse_args()
    
    output_root = Path(args.output_root)
    frame_dir = output_root / 'frames'
    label_dir = output_root / 'labels_auto'
    dataset_root = output_root / 'dataset'
    
    # Step 1: 抽取幀
    if not args.skip_extract:
        image_paths = extract_frames_from_video(
            args.video,
            frame_dir,
            args.num_frames,
            args.extract_method
        )
    else:
        print("[Skip] Frame extraction")
        image_paths = sorted(list(frame_dir.glob('*.jpg')))
    
    # Step 2: 自動標註
    if not args.skip_label:
        auto_label_with_yolo(
            image_paths,
            args.base_weights,
            label_dir,
            args.conf_threshold
        )
        
        print("\n" + "="*60)
        print("⚠️  MANUAL REVIEW REQUIRED")
        print("="*60)
        print("\nPlease review and correct the auto-generated labels:")
        print(f"  Images: {frame_dir}")
        print(f"  Labels: {label_dir}")
        print("\nRecommended tools:")
        print("  - LabelImg: https://github.com/HumanSignal/labelImg")
        print("  - Roboflow: https://roboflow.com")
        print("\nAfter reviewing, re-run with --skip_extract --skip_label")
        
        if not args.skip_finetune:
            response = input("\nContinue to fine-tuning without review? (y/n): ")
            if response.lower() != 'y':
                print("\nStopping here. Please review labels and re-run.")
                return
    else:
        print("[Skip] Auto-labeling")
    
    # Step 3: 創建數據集
    data_yaml = create_finetune_dataset(
        frame_dir,
        label_dir,
        dataset_root,
        args.train_ratio
    )
    
    # Step 4: Fine-tune
    if not args.skip_finetune:
        weights_path = finetune_yolo(
            args.base_weights,
            data_yaml,
            args.finetune_epochs,
            args.finetune_lr,
            args.batch,
            args.device,
            args.project,
            args.name
        )
        
        print("\n" + "="*60)
        print("✓ Complete Pipeline Finished!")
        print("="*60)
        print(f"\n✓ Fine-tuned model: {weights_path}")
        print(f"\nTest it on your video:")
        print(f"  python inference.py --mode demo \\")
        print(f"      --yolo_weights {weights_path} \\")
        print(f"      --video {args.video} \\")
        print(f"      --output_video demo_finetuned.mp4")
    else:
        print("\n[Skip] Fine-tuning")
        print(f"\nDataset prepared: {dataset_root}")
        print(f"Config: {data_yaml}")
        print(f"\nTo fine-tune later:")
        print(f"  python {__file__} ... --skip_extract --skip_label")


if __name__ == "__main__":
    main()