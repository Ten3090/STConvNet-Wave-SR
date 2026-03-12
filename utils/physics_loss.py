#!/usr/bin/env python3
"""
Physics-informed loss functions for ocean wave super-resolution
海浪超分辨率的物理约束损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsInformedLoss(nn.Module):
    """
    Combined loss with physical constraints for ocean wave height prediction
    结合物理约束的海浪有效波高预测损失函数
    """
    
    def __init__(self, 
                 alpha_l1=1.0,           # L1 loss weight
                 alpha_ssim=0.1,         # SSIM loss weight
                 alpha_gradient=0.05,    # Gradient consistency weight
                 alpha_spatial=0.05,     # Spatial smoothness weight
                 alpha_range=0.01,       # Physical range constraint weight
                 alpha_conservation=0.02 # Energy conservation weight
                ):
        super().__init__()
        
        self.alpha_l1 = alpha_l1
        self.alpha_ssim = alpha_ssim
        self.alpha_gradient = alpha_gradient
        self.alpha_spatial = alpha_spatial
        self.alpha_range = alpha_range
        self.alpha_conservation = alpha_conservation
        
        self.l1_loss = nn.L1Loss()
        
    def ssim_loss(self, pred, target, window_size=11):
        """
        SSIM (Structural Similarity Index) loss
        保持结构相似性
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
        mu_target = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
        
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size//2) - mu_pred_sq
        sigma_target_sq = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size//2) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size//2) - mu_pred_target
        
        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        
        return 1 - ssim_map.mean()
    
    def gradient_consistency_loss(self, pred, target):
        """
        Gradient consistency loss - 梯度一致性
        物理约束：波高的空间梯度应该平滑且一致
        """
        # Sobel operators for gradient
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        
        # Calculate gradients
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)
        
        # L1 loss on gradients
        loss_x = F.l1_loss(pred_grad_x, target_grad_x)
        loss_y = F.l1_loss(pred_grad_y, target_grad_y)
        
        return (loss_x + loss_y) / 2
    
    def spatial_smoothness_loss(self, pred):
        """
        Spatial smoothness constraint - 空间平滑性约束
        物理约束：相邻网格点的波高不应该有剧烈变化
        """
        # Calculate differences with neighbors
        diff_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        diff_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        
        # Penalize large differences (but allow some variation)
        # Use L2 to penalize extreme differences more
        loss_x = torch.mean(diff_x ** 2)
        loss_y = torch.mean(diff_y ** 2)
        
        return (loss_x + loss_y) / 2
    
    def physical_range_loss(self, pred):
        """
        Physical range constraint - 物理范围约束
        物理约束：有效波高应该在合理范围内 (通常 0-15m，黑海一般 0-8m)
        """
        # Penalize negative values (physically impossible)
        negative_penalty = torch.relu(-pred).mean()
        
        # Penalize extremely large values (unlikely in Black Sea)
        # Use soft constraint: penalize values > 8m
        extreme_penalty = torch.relu(pred - 8.0).mean()
        
        return negative_penalty + extreme_penalty * 0.1
    
    def energy_conservation_loss(self, pred, target, lr_input):
        """
        Energy conservation constraint - 能量守恒约束
        物理约束：超分辨率后的总能量应该与低分辨率输入一致
        
        波浪能量 ∝ H^2 (H是有效波高)
        """
        # Calculate mean energy in each region
        # pred: [B, 1, 40, 40], lr_input: [B, 4, 8, 8]
        
        # Extract SWH from lr_input (channel 0)
        lr_swh = lr_input[:, 0:1, :, :]  # [B, 1, 8, 8]
        
        # Upsample LR to HR size for comparison
        lr_upsampled = F.interpolate(lr_swh, size=pred.shape[-2:], mode='bicubic', align_corners=True)
        
        # Calculate energy (proportional to H^2)
        pred_energy = (pred ** 2).mean()
        lr_energy = (lr_upsampled ** 2).mean()
        target_energy = (target ** 2).mean()
        
        # Penalize large deviation from input energy
        # But allow some increase (super-resolution should recover details)
        energy_diff = torch.abs(pred_energy - lr_energy)
        
        return energy_diff
    
    def forward(self, pred, target, lr_input=None):
        """
        Combined loss function
        
        Args:
            pred: Predicted HR wave height [B, 1, H, W]
            target: Ground truth HR wave height [B, 1, H, W]
            lr_input: Low-resolution input [B, 4, h, w] (optional, for energy conservation)
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual losses for logging
        """
        # 1. Basic reconstruction loss
        loss_l1 = self.l1_loss(pred, target)
        
        # 2. Structural similarity
        loss_ssim = self.ssim_loss(pred, target)
        
        # 3. Gradient consistency (spatial derivatives should match)
        loss_gradient = self.gradient_consistency_loss(pred, target)
        
        # 4. Spatial smoothness (avoid unrealistic sharp changes)
        loss_spatial = self.spatial_smoothness_loss(pred)
        
        # 5. Physical range constraint (0 < H < 8m for Black Sea)
        loss_range = self.physical_range_loss(pred)
        
        # 6. Energy conservation (if LR input available)
        loss_conservation = 0.0
        if lr_input is not None:
            loss_conservation = self.energy_conservation_loss(pred, target, lr_input)
        
        # Combine all losses
        total_loss = (
            self.alpha_l1 * loss_l1 +
            self.alpha_ssim * loss_ssim +
            self.alpha_gradient * loss_gradient +
            self.alpha_spatial * loss_spatial +
            self.alpha_range * loss_range +
            self.alpha_conservation * loss_conservation
        )
        
        # Return loss dictionary for logging
        loss_dict = {
            'total': total_loss.item(),
            'l1': loss_l1.item(),
            'ssim': loss_ssim.item(),
            'gradient': loss_gradient.item(),
            'spatial': loss_spatial.item(),
            'range': loss_range.item(),
            'conservation': loss_conservation.item() if isinstance(loss_conservation, torch.Tensor) else 0.0
        }
        
        return total_loss, loss_dict


class AdaptivePhysicsLoss(PhysicsInformedLoss):
    """
    Adaptive physics-informed loss with dynamic weight adjustment
    自适应物理约束损失 - 根据训练阶段动态调整权重
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.epoch = 0
        
    def set_epoch(self, epoch):
        """Update epoch for adaptive weighting"""
        self.epoch = epoch
        
        # Early stage: focus on basic reconstruction
        if epoch < 30:
            self.alpha_l1 = 1.0
            self.alpha_ssim = 0.05
            self.alpha_gradient = 0.02
            self.alpha_spatial = 0.01
            self.alpha_range = 0.01
            self.alpha_conservation = 0.01
            
        # Middle stage: gradually increase physics constraints
        elif epoch < 80:
            self.alpha_l1 = 1.0
            self.alpha_ssim = 0.1
            self.alpha_gradient = 0.05
            self.alpha_spatial = 0.05
            self.alpha_range = 0.02
            self.alpha_conservation = 0.02
            
        # Late stage: strong physics constraints for refinement
        else:
            self.alpha_l1 = 1.0
            self.alpha_ssim = 0.15
            self.alpha_gradient = 0.08
            self.alpha_spatial = 0.08
            self.alpha_range = 0.03
            self.alpha_conservation = 0.03
