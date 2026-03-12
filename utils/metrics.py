"""
评估指标
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict


def RMSE(pred: torch.Tensor, target: torch.Tensor) -> float:
    """均方根误差"""
    mse = F.mse_loss(pred, target)
    return torch.sqrt(mse).item()


def MAE(pred: torch.Tensor, target: torch.Tensor) -> float:
    """平均绝对误差"""
    return F.l1_loss(pred, target).item()


def PSNR(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """峰值信噪比"""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return 100.0
    return (20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))).item()


def SSIM(pred: torch.Tensor, target: torch.Tensor, 
         window_size: int = 11, size_average: bool = True) -> float:
    """结构相似性指数
    
    简化版SSIM实现
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=window_size//2)
    mu_target = F.avg_pool2d(target, window_size, stride=1, padding=window_size//2)
    
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target
    
    sigma_pred_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, 
                                 padding=window_size//2) - mu_pred_sq
    sigma_target_sq = F.avg_pool2d(target ** 2, window_size, stride=1,
                                   padding=window_size//2) - mu_target_sq
    sigma_pred_target = F.avg_pool2d(pred * target, window_size, stride=1,
                                     padding=window_size//2) - mu_pred_target
    
    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
               ((mu_pred_sq + mu_target_sq + C1) * 
                (sigma_pred_sq + sigma_target_sq + C2))
    
    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(dim=[1, 2, 3]).item()


def R2_Score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """R²决定系数"""
    target_mean = target.mean()
    ss_tot = ((target - target_mean) ** 2).sum()
    ss_res = ((target - pred) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    return r2.item()


def calculate_metrics(pred: torch.Tensor, 
                     target: torch.Tensor,
                     metrics: list = ['rmse', 'mae', 'psnr', 'ssim', 'r2']) -> Dict[str, float]:
    """计算多个评估指标
    
    Args:
        pred: 预测值 (B, C, H, W)
        target: 真实值 (B, C, H, W)
        metrics: 要计算的指标列表
    Returns:
        results: 指标字典
    """
    results = {}
    
    with torch.no_grad():
        if 'rmse' in metrics:
            results['RMSE'] = RMSE(pred, target)
        
        if 'mae' in metrics:
            results['MAE'] = MAE(pred, target)
        
        if 'psnr' in metrics:
            results['PSNR'] = PSNR(pred, target)
        
        if 'ssim' in metrics:
            results['SSIM'] = SSIM(pred, target)
        
        if 'r2' in metrics:
            results['R2'] = R2_Score(pred, target)
    
    return results



































