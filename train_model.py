# -*- coding: utf-8 -*-
"""
Train D2MP-DDIM Model
完整的訓練腳本，包含 NaN 保護
"""
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
import yaml
import argparse
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from pathlib import Path
from tqdm import tqdm
import warnings

from model import D2MP_DDIM

warnings.filterwarnings('ignore')


# ==================== Dataset ====================

class MotionDataset(Dataset):
    """運動預測數據集"""
    
    def __init__(self, data_root: str, split: str = 'train', n_frames: int = 8, augment: bool = False):
        self.data_root = Path(data_root)
        self.split = split
        self.n_frames = n_frames
        self.augment = augment
        
        self.samples = self._load_data()
        
        print(f"[Dataset] {split} split loaded")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  n_frames: {n_frames}")
    
    def _load_data(self):
        """載入所有序列的數據"""
        split_dir = self.data_root / self.split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        samples = []
        sequences = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        
        for seq_dir in sequences:
            gt_file = seq_dir / "gt" / "gt.txt"
            
            if not gt_file.exists():
                continue
            
            try:
                df = pd.read_csv(
                    gt_file, header=None,
                    names=['frame_id', 'track_id', 'x', 'y', 'w', 'h', 'conf', 'cls', 'vis']
                )
            except:
                continue
            
            for track_id in df.track_id.unique():
                track_df = df[df.track_id == track_id].sort_values('frame_id')
                
                if len(track_df) < self.n_frames + 1:
                    continue
                
                boxes = track_df[['x', 'y', 'w', 'h']].values.astype(np.float32)
                
                if np.isnan(boxes).any() or np.isinf(boxes).any():
                    continue
                
                boxes = np.clip(boxes, 0, None)
                
                for i in range(len(boxes) - self.n_frames):
                    samples.append({
                        'sequence': seq_dir.name,
                        'track_id': int(track_id),
                        'boxes': boxes[i:i + self.n_frames + 1].copy()
                    })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        boxes = sample['boxes'].copy()
        
        if self.augment:
            boxes = self._augment(boxes)
        
        history_boxes = boxes[:self.n_frames]
        target_box = boxes[self.n_frames]
        
        deltas = np.diff(history_boxes, axis=0)
        condition = np.concatenate([history_boxes[1:], deltas], axis=1)
        
        last_box = history_boxes[-1]
        target_motion = target_box - last_box
        
        return {
            'condition': torch.from_numpy(condition).float(),
            'target': torch.from_numpy(target_motion).float()
        }
    
    def _augment(self, boxes):
        """數據增強"""
        if np.random.rand() < 0.3:
            boxes[:, 2:] *= np.random.uniform(0.9, 1.1)
        if np.random.rand() < 0.3:
            boxes[:, :2] += np.random.uniform(-5, 5, size=2)
        if np.random.rand() < 0.2:
            boxes += np.random.normal(0, 1.0, size=boxes.shape)
        
        return np.clip(boxes, 0, None)


# ==================== Trainer ====================

def check_for_nan(tensor, name="tensor"):
    """檢查 NaN/Inf"""
    if torch.isnan(tensor).any():
        print(f"[ERROR] NaN detected in {name}!")
        return True
    if torch.isinf(tensor).any():
        print(f"[ERROR] Inf detected in {name}!")
        return True
    return False


def check_model_health(model):
    """檢查模型健康狀態"""
    for name, param in model.named_parameters():
        if check_for_nan(param.data, f"param:{name}"):
            return False
        if param.grad is not None:
            if check_for_nan(param.grad, f"grad:{name}"):
                return False
    return True


class Trainer:
    """訓練器 (帶 NaN 保護)"""
    
    def __init__(self, config, output_dir, device='cuda'):
        self.config = config
        self.output_dir = Path(output_dir)
        self.device = device
        
        # 創建目錄
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'weights').mkdir(exist_ok=True)
        (self.output_dir / 'logs').mkdir(exist_ok=True)
        
        # 模型
        self.model = D2MP_DDIM(config['model']).to(device)
        
        # 優化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training'].get('weight_decay', 1e-4),
            eps=1e-8
        )
        
        # 學習率調度
        if 'warmup_epochs' in config['training']:
            warmup = config['training']['warmup_epochs']
            total = config['training']['epochs']
            
            def lr_lambda(epoch):
                if epoch < warmup:
                    return (epoch + 1) / warmup
                progress = (epoch - warmup) / (total - warmup)
                return 0.5 * (1 + np.cos(np.pi * progress))
            
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        else:
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['training']['epochs'],
                eta_min=1e-7
            )
        
        # TensorBoard
        self.writer = SummaryWriter(self.output_dir / 'logs')
        
        # 狀態
        self.best_loss = float('inf')
        self.global_step = 0
        self.nan_count = 0
        
        print(f"[Trainer] 初始化完成")
        print(f"  模型參數: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"  學習率: {config['training']['lr']}")
        print(f"  Batch size: {config['training']['batch_size']}")
        print(f"  Epochs: {config['training']['epochs']}")
    
    def train_epoch(self, dataloader, epoch):
        """訓練一個 epoch"""
        self.model.train()
        
        total_loss = 0
        valid_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            condition = batch['condition'].to(self.device)
            target = batch['target'].to(self.device)
            
            # 檢查輸入
            if check_for_nan(condition, "input") or check_for_nan(target, "target"):
                continue
            
            self.optimizer.zero_grad()
            
            try:
                loss = self.model(condition, target)
                
                # 檢查 loss
                if check_for_nan(loss, "loss"):
                    self.nan_count += 1
                    if self.nan_count > 10:
                        print(f"[ERROR] Too many NaN ({self.nan_count})")
                        return None
                    continue
                
                if loss.item() > 100:
                    continue
                
                loss.backward()
                
                # 檢查梯度
                max_grad = 0
                has_nan = False
                for param in self.model.parameters():
                    if param.grad is not None:
                        if check_for_nan(param.grad, "grad"):
                            has_nan = True
                            break
                        max_grad = max(max_grad, param.grad.abs().max().item())
                
                if has_nan:
                    self.optimizer.zero_grad()
                    self.nan_count += 1
                    continue
                
                # 梯度裁剪
                if 'gradient_clip' in self.config['training']:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.optimizer.step()
                
                # 檢查更新後的權重
                if not check_model_health(self.model):
                    print(f"[ERROR] Model unhealthy after update!")
                    return None
                
                total_loss += loss.item()
                valid_batches += 1
                
                # TensorBoard
                self.writer.add_scalar('train/batch_loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/max_grad', max_grad, self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
                
                self.global_step += 1
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg': f'{total_loss/valid_batches:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                })
            
            except RuntimeError as e:
                print(f"[ERROR] {e}")
                self.optimizer.zero_grad()
                continue
        
        if valid_batches == 0:
            return None
        
        return total_loss / valid_batches
    
    def validate(self, dataloader):
        """驗證"""
        self.model.eval()
        
        total_loss = 0
        valid_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                condition = batch['condition'].to(self.device)
                target = batch['target'].to(self.device)
                
                if check_for_nan(condition, "val") or check_for_nan(target, "val"):
                    continue
                
                try:
                    loss = self.model(condition, target)
                    
                    if not check_for_nan(loss, "val_loss") and loss.item() < 100:
                        total_loss += loss.item()
                        valid_batches += 1
                except:
                    continue
        
        if valid_batches == 0:
            return None
        
        return total_loss / valid_batches
    
    def save_checkpoint(self, epoch, loss, is_best=False):
        """保存 checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'train_loss': loss,
            'config': self.config
        }
        
        torch.save(checkpoint, self.output_dir / 'weights' / 'd2mp_ddim_last.pth')
        
        if is_best:
            torch.save(checkpoint, self.output_dir / 'weights' / 'd2mp_ddim_best.pth')
            print(f"[Save] New best model! Loss: {loss:.6f}")
        
        if epoch % self.config['training'].get('save_interval', 10) == 0:
            torch.save(checkpoint, self.output_dir / 'weights' / f'd2mp_ddim_epoch_{epoch}.pth')
    
    def train(self, train_loader, val_loader):
        """完整訓練流程"""
        print("\n" + "="*60)
        print("開始訓練")
        print("="*60)
        
        for epoch in range(1, self.config['training']['epochs'] + 1):
            print(f"\n[Epoch {epoch}/{self.config['training']['epochs']}]")
            
            train_loss = self.train_epoch(train_loader, epoch)
            
            if train_loss is None:
                print("[ERROR] 訓練失敗!")
                break
            
            val_loss = None
            if val_loader and epoch % self.config['training'].get('eval_interval', 5) == 0:
                val_loss = self.validate(val_loader)
            
            self.scheduler.step()
            
            self.writer.add_scalar('train/epoch_loss', train_loss, epoch)
            if val_loss:
                self.writer.add_scalar('val/loss', val_loss, epoch)
            
            print(f"  Train Loss: {train_loss:.6f}")
            if val_loss:
                print(f"  Val Loss: {val_loss:.6f}")
            
            is_best = train_loss < self.best_loss
            if is_best:
                self.best_loss = train_loss
            
            self.save_checkpoint(epoch, train_loss, is_best)
        
        print("\n" + "="*60)
        print("訓練完成!")
        print(f"最佳 Loss: {self.best_loss:.6f}")
        print("="*60)
        
        self.writer.close()


# ==================== Main ====================

def main():
    parser = argparse.ArgumentParser(description="Train D2MP-DDIM")
    
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--output_dir', type=str, default='outputs/ddim', help='Output directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    
    args = parser.parse_args()
    
    # 載入配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "="*60)
    print("D2MP-DDIM Training")
    print("="*60)
    print(f"配置: {args.config}")
    print(f"輸出: {args.output_dir}")
    print(f"設備: {args.device}")
    
    # 數據集
    print("\n[Data] 載入數據集...")
    
    train_dataset = MotionDataset(
        data_root=config['data']['data_root'],
        split=config['data']['train_split'],
        n_frames=config['model']['n_frames'],
        augment=True
    )
    
    val_dataset = MotionDataset(
        data_root=config['data']['data_root'],
        split=config['data']['val_split'],
        n_frames=config['model']['n_frames'],
        augment=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"  訓練集: {len(train_dataset)} 樣本")
    print(f"  驗證集: {len(val_dataset)} 樣本")
    
    # 訓練
    trainer = Trainer(config, args.output_dir, args.device)
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()