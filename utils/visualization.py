"""
可视化工具
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path


def plot_comparison(lr, hr_pred, hr_true, 
                   variable_names=['SWH', 'U10', 'V10'],
                   save_path=None):
    """绘制低分辨率、预测和真实高分辨率的对比图
    
    Args:
        lr: 低分辨率输入 (C, H, W) or (H, W)
        hr_pred: 预测的高分辨率 (C, H, W) or (H, W)
        hr_true: 真实的高分辨率 (C, H, W) or (H, W)
        variable_names: 变量名列表
        save_path: 保存路径
    """
    # 转换为numpy
    if torch.is_tensor(lr):
        lr = lr.cpu().numpy()
    if torch.is_tensor(hr_pred):
        hr_pred = hr_pred.cpu().numpy()
    if torch.is_tensor(hr_true):
        hr_true = hr_true.cpu().numpy()
    
    # 确保是3维 (C, H, W)
    if lr.ndim == 2:
        lr = lr[np.newaxis, :, :]
    if hr_pred.ndim == 2:
        hr_pred = hr_pred[np.newaxis, :, :]
    if hr_true.ndim == 2:
        hr_true = hr_true[np.newaxis, :, :]
    
    num_vars = lr.shape[0]
    
    fig, axes = plt.subplots(num_vars, 4, figsize=(16, 4 * num_vars))
    
    if num_vars == 1:
        axes = axes[np.newaxis, :]
    
    for i in range(num_vars):
        var_name = variable_names[i] if i < len(variable_names) else f'Var {i}'
        
        # 低分辨率
        im0 = axes[i, 0].imshow(lr[i], cmap='viridis')
        axes[i, 0].set_title(f'{var_name} - LR Input')
        axes[i, 0].axis('off')
        plt.colorbar(im0, ax=axes[i, 0])
        
        # 预测
        im1 = axes[i, 1].imshow(hr_pred[i], cmap='viridis')
        axes[i, 1].set_title(f'{var_name} - Predicted HR')
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1])
        
        # 真实
        im2 = axes[i, 2].imshow(hr_true[i], cmap='viridis')
        axes[i, 2].set_title(f'{var_name} - True HR')
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2])
        
        # 误差
        error = np.abs(hr_pred[i] - hr_true[i])
        im3 = axes[i, 3].imshow(error, cmap='Reds')
        axes[i, 3].set_title(f'{var_name} - Absolute Error')
        axes[i, 3].axis('off')
        plt.colorbar(im3, ax=axes[i, 3])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_predictions(predictions, targets, save_dir, prefix='pred'):
    """保存预测结果
    
    Args:
        predictions: 预测结果列表
        targets: 真实值列表
        save_dir: 保存目录
        prefix: 文件名前缀
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        # 保存为numpy数组
        np.save(save_dir / f'{prefix}_pred_{i}.npy', pred.cpu().numpy())
        np.save(save_dir / f'{prefix}_target_{i}.npy', target.cpu().numpy())


def plot_training_history(history, save_path=None):
    """绘制训练历史
    
    Args:
        history: 训练历史字典或JSON文件路径 {'train_loss': [], 'val_loss': [], 'metrics': {}}
        save_path: 保存路径
    """
    # 如果history是路径，先加载JSON
    if isinstance(history, (str, Path)):
        import json
        with open(history, 'r') as f:
            history = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 指标曲线
    if 'metrics' in history:
        for metric_name, values in history['metrics'].items():
            axes[1].plot(values, label=metric_name)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Metric Value')
    axes[1].set_title('Validation Metrics')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    # 测试可视化
    lr = np.random.randn(3, 50, 100)
    hr_pred = np.random.randn(3, 250, 500)
    hr_true = hr_pred + np.random.randn(3, 250, 500) * 0.1
    
    fig = plot_comparison(lr, hr_pred, hr_true)
    plt.show()



































