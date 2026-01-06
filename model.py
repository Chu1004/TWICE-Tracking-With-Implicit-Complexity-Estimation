# -*- coding: utf-8 -*-
"""
D2MP-DDIM: Denoising Diffusion Implicit Models for Motion Prediction (IMPROVED)
修正版本 - 符合論文邏輯
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class HMINet(nn.Module):
    """
    Historical Motion Information Network
    論文 Section 2.3: 6-layer MHSA with class token
    """
    def __init__(self, config):
        super().__init__()
        
        self.hidden_dim = config.get('hidden_dim', 512)
        self.num_layers = config.get('num_layers', 6)
        self.num_heads = config.get('num_heads', 8)
        self.n_frames = config.get('n_frames', 8)
        
        # Input projection: (n_frames-1, 8) -> (n_frames-1, hidden_dim)
        self.input_proj = nn.Linear(8, self.hidden_dim)
        
        # Class token
        self.class_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.n_frames, self.hidden_dim)
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(self.hidden_dim)
    
    def forward(self, condition):
        """
        Args:
            condition: (B, T-1, 8) [boxes, deltas]
        
        Returns:
            encoded: (B, hidden_dim) class token embedding
        """
        B = condition.shape[0]
        
        # Project input
        x = self.input_proj(condition)  # (B, T-1, hidden_dim)
        
        # Add class token
        class_tokens = self.class_token.expand(B, -1, -1)  # (B, 1, hidden_dim)
        x = torch.cat([class_tokens, x], dim=1)  # (B, T, hidden_dim)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :x.shape[1], :]
        
        # Transformer encoding
        x = self.encoder(x)
        
        # Extract class token
        encoded = self.norm(x[:, 0])  # (B, hidden_dim)
        
        return encoded


class MFL(nn.Module):
    """
    Motion Forecasting Layer
    論文中的 Denoising Network ε_θ
    """
    def __init__(self, config):
        super().__init__()
        
        self.hidden_dim = config.get('hidden_dim', 512)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(128, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Motion projection
        self.motion_proj = nn.Linear(4, self.hidden_dim)
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 4)
        )
    
    def get_timestep_embedding(self, timesteps, embedding_dim):
        """
        生成時間步嵌入 (Sinusoidal positional encoding)
        """
        half_dim = embedding_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        
        return emb
    
    def forward(self, motion, timestep, condition_embedding):
        """
        Args:
            motion: (B, 4) noisy motion
            timestep: (B,) timestep
            condition_embedding: (B, hidden_dim) from HMINet
        
        Returns:
            noise_pred: (B, 4) predicted noise
        """
        # Time embedding
        t_emb = self.get_timestep_embedding(timestep.float(), 128)
        t_emb = self.time_embed(t_emb)  # (B, hidden_dim)
        
        # Motion embedding
        m_emb = self.motion_proj(motion)  # (B, hidden_dim)
        
        # Concatenate all embeddings
        combined = torch.cat([m_emb, t_emb, condition_embedding], dim=1)
        
        # Predict noise
        noise_pred = self.fusion(combined)
        
        return noise_pred


class ComplexityEstimator(nn.Module):
    """
    論文 Section 2.3: Complexity Estimator
    C = 0.5·σ²_a + 0.3·Δθ + 0.2·Δs
    """
    def __init__(self):
        super().__init__()
    
    def compute_acceleration_variance(self, history_boxes):
        """計算加速度方差 σ²_a"""
        velocities = history_boxes[:, 1:] - history_boxes[:, :-1]
        
        if velocities.shape[1] > 1:
            accelerations = velocities[:, 1:] - velocities[:, :-1]
            acc_xy = accelerations[:, :, :2]
            variance = torch.var(acc_xy, dim=1).mean(dim=1)
            return variance
        else:
            return torch.zeros(history_boxes.shape[0], device=history_boxes.device)
    
    def compute_direction_change(self, history_boxes):
        """計算方向變化 Δθ"""
        velocities = history_boxes[:, 1:] - history_boxes[:, :-1]
        vel_xy = velocities[:, :, :2]
        
        if vel_xy.shape[1] > 1:
            v1 = vel_xy[:, :-1]
            v2 = vel_xy[:, 1:]
            
            v1_norm = F.normalize(v1, dim=2, eps=1e-6)
            v2_norm = F.normalize(v2, dim=2, eps=1e-6)
            
            cos_theta = (v1_norm * v2_norm).sum(dim=2)
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            
            theta = torch.acos(cos_theta)
            avg_theta = theta.mean(dim=1)
            
            return avg_theta
        else:
            return torch.zeros(history_boxes.shape[0], device=history_boxes.device)
    
    def compute_size_change(self, history_boxes):
        """計算尺寸變化 Δs"""
        sizes = history_boxes[:, :, 2:4]
        areas = sizes[:, :, 0] * sizes[:, :, 1]
        
        if areas.shape[1] > 1:
            area_changes = torch.abs(areas[:, 1:] - areas[:, :-1]) / (areas[:, :-1] + 1e-6)
            avg_change = area_changes.mean(dim=1)
            return avg_change
        else:
            return torch.zeros(history_boxes.shape[0], device=history_boxes.device)
    
    def forward(self, history_boxes):
        """
        Args:
            history_boxes: (B, T, 4)
        
        Returns:
            complexity: (B,) 範圍 [0, 1]
            steps: (B,) 推薦的 DDIM 步數
        """
        sigma_a = self.compute_acceleration_variance(history_boxes)
        delta_theta = self.compute_direction_change(history_boxes)
        delta_s = self.compute_size_change(history_boxes)
        
        # 歸一化
        sigma_a_norm = torch.sigmoid(sigma_a)
        delta_theta_norm = delta_theta / (np.pi + 1e-6)
        delta_s_norm = torch.clamp(delta_s, 0, 1)
        
        # 論文公式
        complexity = 0.5 * sigma_a_norm + 0.3 * delta_theta_norm + 0.2 * delta_s_norm
        
        # 決定步數
        steps = torch.ones_like(complexity, dtype=torch.long)
        steps[complexity >= 0.3] = 2
        steps[complexity >= 0.7] = 3
        
        return complexity, steps


class DDIMScheduler:
    """
    DDIM Noise Scheduler
    論文 Section 2.2: ODE-based deterministic sampling
    """
    def __init__(self, num_train_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_train_timesteps = num_train_timesteps
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # For decoupled diffusion (D²MP compatibility)
        # Forward: M_{f,t} = M_{f,0} + t*c + √t·z
        # where c is a learnable drift term (implicitly in ε_θ)
    
    def add_noise(self, original, noise, timesteps):
        """
        前向過程: 添加噪聲
        論文中的 Forward Process: M_{f,t} = √α_t·M_{f,0} + √(1-α_t)·z
        """
        alphas_cumprod = self.alphas_cumprod.to(original.device)
        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(original.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
        
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(original.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
        
        noisy = sqrt_alpha_prod * original + sqrt_one_minus_alpha_prod * noise
        return noisy


class D2MP_DDIM(nn.Module):
    """
    完整的 D2MP-DDIM 模型 (改進版)
    論文 Section 2.2 & 2.3
    
    主要改進:
    1. 修正 DDIM 採樣公式
    2. 統一與 D²MP 的關係 (當 num_steps=1 時等同於 D²MP)
    3. 改進 adaptive sampling 的輸入處理
    """
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.n_frames = config.get('n_frames', 8)
        
        # Encoder
        self.encoder = HMINet(config)
        
        # Denoising network
        self.denoise_net = MFL(config)
        
        # Complexity estimator
        self.complexity_estimator = ComplexityEstimator()
        
        # Noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=config.get('num_train_timesteps', 1000),
            beta_start=config.get('beta_start', 0.0001),
            beta_end=config.get('beta_end', 0.02)
        )
        
        # Adaptive sampling
        self.use_adaptive = config.get('adaptive', False)
    
    def forward(self, condition, target):
        """
        訓練時的前向傳播
        
        Args:
            condition: (B, T-1, 8)
            target: (B, 4) ground truth motion
        
        Returns:
            loss: MSE loss
        """
        batch_size = condition.shape[0]
        device = condition.device
        
        # Encode condition
        encoded = self.encoder(condition)
        
        # Sample timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.num_train_timesteps,
            (batch_size,), device=device
        ).long()
        
        # Add noise
        noise = torch.randn_like(target)
        noisy_target = self.noise_scheduler.add_noise(target, noise, timesteps)
        
        # Predict noise
        noise_pred = self.denoise_net(noisy_target, timesteps, encoded)
        
        # Loss
        loss = F.mse_loss(noise_pred, noise)
        
        return loss
    
    @torch.no_grad()
    def sample_ddim(self, condition, device='cuda', num_steps=1, seed=None):
        """
        論文 Section 2.2: DDIM 採樣 (修正版)
        
        CRITICAL FIX: 正確的 DDIM 更新公式
        論文公式: μ_θ = √α_{t-Δt} · [M_t - √(1-α_t)·ε_θ] / √α_t + √(1-α_{t-Δt})·ε_θ
        
        當 num_steps=1 時，等同於 D²MP 的單步採樣
        
        Args:
            condition: (B, T-1, 8) 條件輸入
            device: 設備
            num_steps: DDIM 步數
            seed: 隨機種子 (用於確保確定性)
        """
        batch_size = condition.shape[0]
        
        # Encode
        encoded = self.encoder(condition)
        
        # Initialize from noise
        # FIXED: 使用固定種子確保確定性
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
            x_t = torch.randn(batch_size, 4, device=device, generator=generator)
        else:
            x_t = torch.randn(batch_size, 4, device=device)
        
        # DDIM timesteps
        timesteps = torch.linspace(
            self.noise_scheduler.num_train_timesteps - 1,
            0,
            num_steps,
            device=device
        ).long()
        
        # Iterative denoising
        for i, t in enumerate(timesteps):
            t_batch = t.unsqueeze(0).repeat(batch_size)
            
            # Predict noise
            noise_pred = self.denoise_net(x_t, t_batch, encoded)
            
            # Get alpha values - FIXED: use .item() to get CPU index
            t_idx = t.item()
            alpha_t = self.noise_scheduler.alphas_cumprod[t_idx].to(device)
            
            if i < len(timesteps) - 1:
                # Not the last step
                t_prev = timesteps[i + 1]
                t_prev_idx = t_prev.item()
                alpha_t_prev = self.noise_scheduler.alphas_cumprod[t_prev_idx].to(device)
            else:
                # Last step: go to t=0
                alpha_t_prev = torch.tensor(1.0, device=device)
            
            # DDIM update (CORRECTED)
            # 論文公式: μ_θ = √α_{t-Δt} · [M_t - √(1-α_t)·ε_θ] / √α_t + √(1-α_{t-Δt})·ε_θ
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            sqrt_one_minus_alpha_t_prev = torch.sqrt(1 - alpha_t_prev)
            
            # Predicted x_0
            x_0_pred = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
            
            # Direction pointing to x_t (這是關鍵修正)
            # x_{t-1} = √α_{t-1} · x_0 + √(1-α_{t-1}) · ε_θ
            x_t = sqrt_alpha_t_prev * x_0_pred + sqrt_one_minus_alpha_t_prev * noise_pred
        
        return x_t
    
    @torch.no_grad()
    def sample_d2mp(self, condition, device='cuda', seed=None):
        """
        D²MP 單步採樣 (for comparison)
        論文 Section 2.1: M_{f,0} = -c_θ(M_{f,t}, t, C_f)
        
        這應該等同於 sample_ddim with num_steps=1
        
        Args:
            condition: (B, T-1, 8) 條件輸入
            device: 設備
            seed: 隨機種子 (用於與 DDIM 比較)
        """
        batch_size = condition.shape[0]
        
        # Encode
        encoded = self.encoder(condition)
        
        # Sample from noise
        # FIXED: 使用固定種子確保與 DDIM 比較時使用相同初始噪聲
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(seed)
            x_t = torch.randn(batch_size, 4, device=device, generator=generator)
        else:
            x_t = torch.randn(batch_size, 4, device=device)
        
        # Single timestep (can be random or fixed)
        # FIXED: 使用 CPU 索引來避免設備不匹配
        t_idx = self.noise_scheduler.num_train_timesteps - 1
        t_batch = torch.tensor([t_idx] * batch_size, device=device)
        
        # Predict noise
        noise_pred = self.denoise_net(x_t, t_batch, encoded)
        
        # D²MP: M_{f,0} = -c_θ ≈ predicted clean sample
        alpha_t = self.noise_scheduler.alphas_cumprod[t_idx].to(device)
        sqrt_alpha_t = torch.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
        
        # Predict x_0 directly
        x_0_pred = (x_t - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
        
        return x_0_pred
    
    @torch.no_grad()
    def sample_adaptive(self, condition, device='cuda'):
        """
        論文 Section 2.3: Adaptive Sampling (修正版)
        根據運動複雜度動態調整步數
        
        CRITICAL FIX: 從 condition 正確提取 history_boxes
        """
        if not self.use_adaptive:
            return self.sample_ddim(condition, device, num_steps=1), None, None
        
        batch_size = condition.shape[0]
        
        # FIXED: 從 condition (B, T-1, 8) 正確提取歷史 boxes
        # condition = [boxes[1:], deltas] concatenated along last dim
        # 所以 boxes 在前 4 個維度
        history_boxes_partial = condition[:, :, :4]  # (B, T-1, 4)
        
        # 我們需要重建完整的 history (T frames)
        # 使用 deltas 反推第一個 box
        deltas = condition[:, :, 4:]  # (B, T-1, 4)
        
        # 構建完整歷史: [first_box, boxes[1:]]
        # first_box = boxes[1] - deltas[0]
        first_box = history_boxes_partial[:, 0:1, :] - deltas[:, 0:1, :]  # (B, 1, 4)
        history_boxes = torch.cat([first_box, history_boxes_partial], dim=1)  # (B, T, 4)
        
        # 估計複雜度
        complexity, adaptive_steps = self.complexity_estimator(history_boxes)
        
        # 對每個樣本使用不同步數 (這裡可以優化為批次處理)
        results = []
        
        for i in range(batch_size):
            num_steps = adaptive_steps[i].item()
            sample = self.sample_ddim(
                condition[i:i+1],
                device=device,
                num_steps=num_steps
            )
            results.append(sample)
        
        output = torch.cat(results, dim=0)
        
        return output, complexity, adaptive_steps
    
    @torch.no_grad()
    def sample(self, condition, device='cuda', sampler='ddim', seed=None):
        """
        統一的採樣介面
        
        Args:
            condition: (B, T-1, 8)
            device: 'cuda' or 'cpu'
            sampler: 'ddim', 'd2mp', or 'adaptive'
            seed: 隨機種子 (可選，用於確定性採樣)
        
        Returns:
            motion: (B, 4) predicted motion
        """
        if sampler == 'adaptive' and self.use_adaptive:
            motion, _, _ = self.sample_adaptive(condition, device)
        elif sampler == 'd2mp':
            motion = self.sample_d2mp(condition, device, seed=seed)
        else:
            motion = self.sample_ddim(condition, device, num_steps=1, seed=seed)
        
        return motion


def test_model():
    """測試模型 - 驗證改進"""
    print("="*60)
    print("測試 D2MP-DDIM 模型 (改進版)")
    print("="*60)
    
    config = {
        'hidden_dim': 512,
        'num_layers': 6,
        'num_heads': 8,
        'n_frames': 8,
        'num_train_timesteps': 1000,
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'adaptive': True
    }
    
    model = D2MP_DDIM(config).cuda()
    model.eval()
    
    print(f"模型參數: {sum(p.numel() for p in model.parameters()):,}")
    
    # 測試前向傳播
    batch_size = 4
    condition = torch.randn(batch_size, 7, 8).cuda()
    target = torch.randn(batch_size, 4).cuda()
    
    print(f"\n輸入形狀:")
    print(f"  condition: {condition.shape}")
    print(f"  target: {target.shape}")
    
    # 訓練模式
    model.train()
    loss = model(condition, target)
    print(f"\n訓練 Loss: {loss.item():.6f}")
    
    # 推理模式
    model.eval()
    
    # 測試不同採樣器
    print("\n測試不同採樣器:")
    
    # DDIM 單步 (無種子 - 隨機)
    motion_ddim1 = model.sample(condition, device='cuda', sampler='ddim')
    print(f"  DDIM (1-step): {motion_ddim1.shape}")
    
    # DDIM 多步
    motion_ddim3 = model.sample_ddim(condition, device='cuda', num_steps=3, seed=42)
    print(f"  DDIM (3-step): {motion_ddim3.shape}")
    
    # D²MP (使用相同種子進行比較)
    motion_d2mp = model.sample_d2mp(condition, device='cuda', seed=42)
    print(f"  D²MP: {motion_d2mp.shape}")
    
    # Adaptive
    motion_adap, complexity, steps = model.sample_adaptive(condition, device='cuda')
    print(f"  Adaptive: {motion_adap.shape}")
    print(f"    複雜度: {complexity}")
    print(f"    步數: {steps}")
    
    # 驗證 DDIM 1-step 與 D²MP 的一致性 (使用相同種子)
    print(f"\n驗證 DDIM 1-step ≈ D²MP (使用相同初始噪聲):")
    motion_ddim1_seed = model.sample_ddim(condition, device='cuda', num_steps=1, seed=42)
    motion_d2mp_seed = model.sample_d2mp(condition, device='cuda', seed=42)
    diff = (motion_ddim1_seed - motion_d2mp_seed).abs().mean()
    print(f"  平均差異: {diff.item():.6f}")
    if diff.item() < 0.1:
        print(f"  結論: ✓ DDIM 1-step 與 D²MP 高度一致")
    else:
        print(f"  結論: ⚠ 存在較大差異 (可能是公式實作的微小差異)")
    
    # 測試確定性 (DDIM 應該是確定的 - 使用相同種子)
    print(f"\n測試確定性 (DDIM with fixed seed):")
    motion_test1 = model.sample_ddim(condition, device='cuda', num_steps=1, seed=123)
    motion_test2 = model.sample_ddim(condition, device='cuda', num_steps=1, seed=123)
    deterministic_diff = (motion_test1 - motion_test2).abs().max()
    print(f"  相同種子兩次採樣的最大差異: {deterministic_diff.item():.8f}")
    print(f"  是否確定性: {'✓ 完全確定' if deterministic_diff < 1e-6 else '✗ 不確定'}")
    
    # 測試不同種子產生不同結果
    print(f"\n測試隨機性 (不同種子應產生不同結果):")
    motion_seed1 = model.sample_ddim(condition, device='cuda', num_steps=1, seed=1)
    motion_seed2 = model.sample_ddim(condition, device='cuda', num_steps=1, seed=2)
    random_diff = (motion_seed1 - motion_seed2).abs().mean()
    print(f"  不同種子採樣的平均差異: {random_diff.item():.6f}")
    print(f"  是否有隨機性: {'✓ 正常' if random_diff > 0.01 else '⚠ 種子可能無效'}")
    
    print("\n" + "="*60)
    print("測試總結:")
    print("="*60)
    print("✓ 1. 模型前向傳播正常")
    print("✓ 2. 所有採樣器運行正常")
    print(f"{'✓' if deterministic_diff < 1e-6 else '✗'} 3. DDIM 確定性: {'通過' if deterministic_diff < 1e-6 else '失敗'}")
    print(f"{'✓' if diff < 0.1 else '⚠'} 4. DDIM ≈ D²MP: {'一致' if diff < 0.1 else '有差異'}")
    print(f"{'✓' if random_diff > 0.01 else '⚠'} 5. 隨機種子機制: {'正常' if random_diff > 0.01 else '異常'}")
    print("="*60)
    print("\n✓ 模型測試完成!")



if __name__ == "__main__":
    test_model()