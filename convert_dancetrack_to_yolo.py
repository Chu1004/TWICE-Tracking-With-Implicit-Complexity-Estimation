import os
import shutil
from pathlib import Path
from PIL import Image

def convert_dancetrack_to_yolo(root_dir, output_dir):
    """
    Convert DanceTrack dataset to YOLO11 format
    
    DanceTrack GT format: frame_id, track_id, x, y, w, h, conf, x_3d, y_3d
    YOLO format: class_id x_center y_center width height (normalized 0-1)
    
    Args:
        root_dir: Path to DanceTrack dataset root
        output_dir: Path to output YOLO format dataset
    """
    
    root_path = Path(root_dir)
    output_path = Path(output_dir)
    
    # Create output directory structure
    for split in ['train', 'val']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ['train', 'val']:
        split_dir = root_path / split
        
        if not split_dir.exists():
            print(f"Warning: {split_dir} does not exist, skipping...")
            continue
        
        # Process each sequence
        for seq_dir in sorted(split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            
            print(f"Processing {split}/{seq_dir.name}...")
            
            img_dir = seq_dir / 'img1'
            gt_file = seq_dir / 'gt' / 'gt.txt'
            
            if not gt_file.exists():
                print(f"Warning: GT file not found for {seq_dir.name}")
                continue
            
            # Read GT annotations
            annotations = {}
            with open(gt_file, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) < 6:
                        continue
                    
                    frame_id = int(parts[0])
                    track_id = int(parts[1])
                    x = float(parts[2])
                    y = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])
                    
                    if frame_id not in annotations:
                        annotations[frame_id] = []
                    
                    annotations[frame_id].append({
                        'track_id': track_id,
                        'bbox': [x, y, w, h]
                    })
            
            # Process each frame
            if not img_dir.exists():
                print(f"Warning: Image directory not found for {seq_dir.name}")
                continue
            
            for img_file in sorted(img_dir.glob('*.jpg')):
                frame_id = int(img_file.stem)
                
                # Get image dimensions
                try:
                    img = Image.open(img_file)
                    img_width, img_height = img.size
                except Exception as e:
                    print(f"Error reading image {img_file}: {e}")
                    continue
                
                # Copy image to output directory
                output_img_path = output_path / split / 'images' / f"{seq_dir.name}_{frame_id:08d}.jpg"
                shutil.copy2(img_file, output_img_path)
                
                # Create label file
                output_label_path = output_path / split / 'labels' / f"{seq_dir.name}_{frame_id:08d}.txt"
                
                with open(output_label_path, 'w') as f:
                    if frame_id in annotations:
                        for ann in annotations[frame_id]:
                            x, y, w, h = ann['bbox']
                            
                            # Convert to YOLO format (center x, center y, width, height) normalized
                            x_center = (x + w / 2) / img_width
                            y_center = (y + h / 2) / img_height
                            w_norm = w / img_width
                            h_norm = h / img_height
                            
                            # Class ID is 0 for person (single class)
                            class_id = 0
                            
                            # Write in YOLO format
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    
    # Create dataset.yaml for YOLO
    yaml_content = f"""# DanceTrack Dataset in YOLO Format
path: {output_path.absolute()}
train: train/images
val: val/images

# Classes
names:
  0: person

# Number of classes
nc: 1
"""
    
    with open(output_path / 'dataset.yaml', 'w') as f:
        f.write(yaml_content)
    
    print(f"\nConversion completed!")
    print(f"Output directory: {output_path}")
    print(f"Dataset configuration saved to: {output_path / 'dataset.yaml'}")


if __name__ == "__main__":
    # Example usage
    root_dir = r"C:\Users\User\Desktop\AIFP\pipeline\dancetrack"
    output_dir = r"C:\Users\User\Desktop\AIFP\pipeline\dancetrack_yolo"
    
    convert_dancetrack_to_yolo(root_dir, output_dir)
    
    print("\n" + "="*60)
    print("YOLO Dataset Structure:")
    print("="*60)
    print("""
dancetrack_yolo/
    train/
        images/
            dancetrack0001_00000001.jpg
            dancetrack0001_00000002.jpg
            ...
        labels/
            dancetrack0001_00000001.txt
            dancetrack0001_00000002.txt
            ...
    val/
        images/
            dancetrack0005_00000001.jpg
            ...
        labels/
            dancetrack0005_00000001.txt
            ...
    dataset.yaml
    """)