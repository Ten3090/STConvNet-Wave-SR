"""
Spatio-Temporal Dataset for Ocean Wave Super-Resolution
时空数据集 - 返回连续时间帧的序列

基于 Yearly5xDataset 扩展，支持时间序列输入
"""

import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
import json
from pathlib import Path


class SpatioTemporalDataset(Dataset):
    """
    时空超分辨率数据集
    
    返回连续 T 个时刻的 LR 数据和中间时刻的 HR 数据
    
    Args:
        lr_path: Path to LR NetCDF (swh, period, dir_sin, dir_cos)
        hr_path: Path to HR NetCDF (swh only)
        num_frames: 时间帧数（默认 5）
        stats_path: Path to stats JSON
        normalize: Whether to normalize
    """
    
    def __init__(self, lr_path, hr_path, num_frames=5, stats_path=None, normalize=True):
        super().__init__()
        
        self.ds_lr = xr.open_dataset(lr_path)
        self.ds_hr = xr.open_dataset(hr_path)
        self.num_frames = num_frames
        self.half_frames = num_frames // 2
        self.normalize = normalize
        
        # 总时间步数
        total_timesteps = len(self.ds_lr.time)
        
        # 可用样本数（需要连续 num_frames 个时刻）
        self.n_samples = total_timesteps - num_frames + 1
        
        # Load stats
        if stats_path and Path(stats_path).exists():
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)
        else:
            self.stats = None
        
        print(f'SpatioTemporalDataset loaded:')
        print(f'  Total timesteps: {total_timesteps}')
        print(f'  Num frames: {num_frames}')
        print(f'  Usable samples: {self.n_samples}')
        print(f'  LR shape per frame: {self.ds_lr.swh.isel(time=0).shape} (4 channels)')
        print(f'  HR shape per frame: {self.ds_hr.swh.isel(time=0).shape} (1 channel)')
        if self.stats:
            print(f'  Stats loaded from {stats_path}')
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        返回连续 T 个时刻的 LR 和最后时刻的 HR (因果模型)
        
        Args:
            idx: 起始索引
        
        Returns:
            lr_seq: [T, 4, H, W] - T 个时刻的 LR 数据 [idx, idx+1, ..., idx+T-1]
            hr: [1, H, W] - 最后时刻 (idx+T-1) 的 HR 数据
        
        Example:
            num_frames=5, idx=100
            lr_seq: [100, 101, 102, 103, 104] (5 帧历史)
            hr: 104 (最后一帧，当前时刻)
        """
        # 加载连续 T 个时刻的 LR
        lr_seq = []
        for t in range(self.num_frames):
            time_idx = idx + t
            
            # LR: 4 channels (swh, period, dir_sin, dir_cos)
            lr_swh = self.ds_lr.swh.isel(time=time_idx).values
            lr_period = self.ds_lr.period.isel(time=time_idx).values
            lr_dir_sin = self.ds_lr.dir_sin.isel(time=time_idx).values
            lr_dir_cos = self.ds_lr.dir_cos.isel(time=time_idx).values
            
            # Stack to (4, H, W)
            lr = np.stack([lr_swh, lr_period, lr_dir_sin, lr_dir_cos], axis=0)
            
            # Handle NaN
            lr = np.nan_to_num(lr, nan=0.0)
            
            # Normalize each channel separately
            if self.normalize and self.stats:
                for i, var in enumerate(['swh', 'period', 'dir_sin', 'dir_cos']):
                    mean = self.stats[var]['lr_mean']
                    std = self.stats[var]['lr_std']
                    lr[i] = (lr[i] - mean) / (std + 1e-8)
            
            lr_seq.append(lr)
        
        # Stack to [T, 4, H, W]
        lr_seq = np.stack(lr_seq, axis=0)
        
        # 加载最后时刻的 HR (因果模型)
        target_idx = idx + self.num_frames - 1
        hr = self.ds_hr.swh.isel(time=target_idx).values[np.newaxis, :, :]
        
        # Handle NaN
        hr = np.nan_to_num(hr, nan=0.0)
        
        # Normalize HR
        if self.normalize and self.stats:
            hr_mean = self.stats['swh']['hr_mean']
            hr_std = self.stats['swh']['hr_std']
            hr = (hr - hr_mean) / (hr_std + 1e-8)
        
        # Convert to tensor
        lr_seq = torch.from_numpy(lr_seq).float()
        hr = torch.from_numpy(hr).float()
        
        return lr_seq, hr


def create_spatiotemporal_dataloaders(
    data_dir='data/yearly_hr_0p1',
    num_frames=5,
    batch_size=8,
    num_workers=4,
    normalize=True
):
    """
    创建时空超分辨率数据加载器
    
    Args:
        data_dir: 数据目录
        num_frames: 时间帧数（默认 5）
        batch_size: 批大小
        num_workers: 工作进程数
        normalize: 是否归一化
    
    Returns:
        train_loader, val_loader, test_loader
    """
    data_dir = Path(data_dir)
    
    train_lr = data_dir / 'train_lr.nc'
    train_hr = data_dir / 'train_hr.nc'
    val_lr = data_dir / 'val_lr.nc'
    val_hr = data_dir / 'val_hr.nc'
    test_lr = data_dir / 'test_lr.nc'
    test_hr = data_dir / 'test_hr.nc'
    stats_file = data_dir / 'stats.json'
    
    print('\n' + '='*70)
    print(f'Loading Spatio-Temporal Dataset (T={num_frames} frames)')
    print('='*70)
    print(f'Data directory: {data_dir}')
    print(f'LR: 8x8 (swh, period, dir_sin, dir_cos) × {num_frames} frames')
    print(f'HR: 40x40 (swh) - middle frame')
    print('='*70 + '\n')
    
    train_dataset = SpatioTemporalDataset(
        str(train_lr), str(train_hr),
        num_frames=num_frames,
        stats_path=str(stats_file) if stats_file.exists() else None,
        normalize=normalize
    )
    print(f'✓ Train: {len(train_dataset)} samples\n')
    
    val_dataset = SpatioTemporalDataset(
        str(val_lr), str(val_hr),
        num_frames=num_frames,
        stats_path=str(stats_file) if stats_file.exists() else None,
        normalize=normalize
    )
    print(f'✓ Val: {len(val_dataset)} samples\n')
    
    test_dataset = SpatioTemporalDataset(
        str(test_lr), str(test_hr),
        num_frames=num_frames,
        stats_path=str(stats_file) if stats_file.exists() else None,
        normalize=normalize
    )
    print(f'✓ Test: {len(test_dataset)} samples\n')
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    print('='*70)
    print('DataLoaders created:')
    print(f'  Train batches: {len(train_loader)}')
    print(f'  Val batches: {len(val_loader)}')
    print(f'  Test batches: {len(test_loader)}')
    print(f'  Batch size: {batch_size}')
    print(f'  Output shape: [B, T={num_frames}, 4, 8, 8] (LR), [B, 1, 40, 40] (HR)')
    print('='*70 + '\n')
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # 测试数据加载器
    print("Testing SpatioTemporalDataset...")
    
    train_loader, val_loader, test_loader = create_spatiotemporal_dataloaders(
        data_dir='data/yearly_hr_0p1',
        num_frames=5,
        batch_size=4,
        num_workers=0  # 测试时使用 0
    )
    
    # 测试一个批次
    for lr_seq, hr in train_loader:
        print(f"\nBatch shapes:")
        print(f"  LR sequence: {lr_seq.shape}")  # [B, T, 4, 8, 8]
        print(f"  HR target: {hr.shape}")  # [B, 1, 40, 40]
        print(f"\nLR sequence stats:")
        print(f"  Min: {lr_seq.min():.4f}")
        print(f"  Max: {lr_seq.max():.4f}")
        print(f"  Mean: {lr_seq.mean():.4f}")
        print(f"\nHR target stats:")
        print(f"  Min: {hr.min():.4f}")
        print(f"  Max: {hr.max():.4f}")
        print(f"  Mean: {hr.mean():.4f}")
        break
    
    print("\n[SUCCESS] Dataset test passed!")
