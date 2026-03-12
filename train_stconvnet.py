"""
Training script for STConvNet-Wave-SR.

This script trains the ConvLSTM + partial-fusion Swin Transformer model
for 5x significant wave height super-resolution.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.optim as optim
import json
import random
import numpy as np

from models.spatiotemporal_partial_fusion_sr import create_spatiotemporal_partial_fusion_sr
from data.spatiotemporal_dataset import create_spatiotemporal_dataloaders
from utils.metrics import calculate_metrics
from utils.visualization import plot_training_history
from utils.physics_loss import AdaptivePhysicsLoss


class WarmupCosineScheduler:
    """Cosine scheduler with warmup."""
    def __init__(self, optimizer, warmup_epochs, max_epochs, 
                 eta_min=0, last_epoch=-1):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.eta_min = eta_min
        self.base_lr = optimizer.param_groups[0]['lr']
        self.current_epoch = last_epoch + 1
        
    def step(self):
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            progress = (self.current_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.eta_min + (self.base_lr - self.eta_min) * \
                 0.5 * (1 + torch.cos(torch.tensor(progress * 3.141592653589793)))
            lr = float(lr)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr


def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    
    for batch_idx, (lr_seq, hr_batch) in enumerate(loader):
        lr_seq, hr_batch = lr_seq.to(device), hr_batch.to(device)
        
        optimizer.zero_grad()
        pred = model(lr_seq)
        loss, loss_dict = criterion(pred, hr_batch, lr_seq[:, -1])
        
        if torch.isnan(loss):
            print(f"\n[警告] 检测到NaN损失，跳过该批次")
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_metrics = []
    
    with torch.no_grad():
        for lr_seq, hr_batch in loader:
            lr_seq, hr_batch = lr_seq.to(device), hr_batch.to(device)
            
            pred = model(lr_seq)
            loss, loss_dict = criterion(pred, hr_batch, lr_seq[:, -1])
            total_loss += loss.item()
            
            metrics = calculate_metrics(pred, hr_batch)
            all_metrics.append(metrics)
    
    avg_loss = total_loss / len(loader)
    avg_metrics = {k: sum(m[k] for m in all_metrics) / len(all_metrics) 
                   for k in all_metrics[0].keys()}
    
    return avg_loss, avg_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='stconvnet_release', help='Experiment name')
    parser.add_argument('--data_dir', type=str, default='data/yearly_hr_0p1', help='Path to processed dataset')
    parser.add_argument('--device', type=str, default='cuda', help='Training device')
    args = parser.parse_args()

    fusion_stages = [1, 2, 3, 4]
    
    config = {
        'experiment_id': args.name,
        'experiment_name': 'STConvNet-Wave-SR',
        'baseline': 'full_multiscale',
        
        'model': 'SpatioTemporalPartialFusionSR',
        'num_frames': 5,
        'fusion_stages': fusion_stages,
        'use_multiscale': True,
        'use_temporal': True,
        'temporal_dim': 64,
        
        'epochs': 150,
        'batch_size': 16,
        'learning_rate': 3e-4,
        'weight_decay': 1e-4,
        'warmup_epochs': 10,
        'min_lr': 1e-7,
        
        'loss': 'AdaptivePhysicsLoss',
        'alpha_l1': 1.0,
        'alpha_ssim': 0.03,
        'alpha_gradient': 0.01,
        'alpha_spatial': 0.01,
        'alpha_range': 0.005,
        'alpha_conservation': 0.005,
        
        'data_dir': args.data_dir,
        'num_workers': 4,
        
        'output_dir': f'experiments/exp13/{args.name}',
        'save_freq': 10,
        
        'device': args.device,
        'seed': 42
    }
    
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    set_seed(config['seed'])
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print(f"实验名称: {config['experiment_name']}")
    print(f"实验ID: {config['experiment_id']}")
    print("="*80)
    print(f"设备: {device}")
    print(f"训练轮数: {config['epochs']}")
    print(f"学习率: {config['learning_rate']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"时序帧数: {config['num_frames']}")
    print(f"融合stages: {config['fusion_stages']}")
    print(f"时序维度: {config['temporal_dim']}")
    print(f"Warmup epochs: {config['warmup_epochs']}")
    print(f"损失函数: AdaptivePhysicsLoss")
    print("="*80)
    
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(exist_ok=True)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n加载时空数据...")
    train_loader, val_loader, test_loader = create_spatiotemporal_dataloaders(
        data_dir=config['data_dir'],
        num_frames=config['num_frames'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    fusion_info = f"Stage {'+'.join(map(str, config['fusion_stages']))}"
    print(f"\nCreating model with {fusion_info}...")
    model = create_spatiotemporal_partial_fusion_sr(
        num_frames=config['num_frames'],
        fusion_stages=config['fusion_stages'],
        use_temporal=config['use_temporal']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    criterion = AdaptivePhysicsLoss(
        alpha_l1=config['alpha_l1'],
        alpha_ssim=config['alpha_ssim'],
        alpha_gradient=config['alpha_gradient'],
        alpha_spatial=config['alpha_spatial'],
        alpha_range=config['alpha_range'],
        alpha_conservation=config['alpha_conservation']
    )
    print(f"\nLoss function: AdaptivePhysicsLoss")
    
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=config['warmup_epochs'],
        max_epochs=config['epochs'],
        eta_min=config['min_lr']
    )
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'metrics': {'RMSE': [], 'MAE': [], 'PSNR': [], 'SSIM': [], 'R2': []}
    }
    
    best_rmse = float('inf')
    best_epoch = 0
    
    print(f"\n开始训练...")
    print("="*80)
    
    for epoch in range(1, config['epochs'] + 1):
        if hasattr(criterion, 'set_epoch'):
            criterion.set_epoch(epoch)
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        current_lr = scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)
        for k in val_metrics:
            history['metrics'][k].append(val_metrics[k])
        
        print(f"\nEpoch {epoch}/{config['epochs']} | Stages {config['fusion_stages']}")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"RMSE: {val_metrics['RMSE']:.4f}m, MAE: {val_metrics['MAE']:.4f}m")
        print(f"PSNR: {val_metrics['PSNR']:.2f}dB, SSIM: {val_metrics['SSIM']:.4f}")
        print(f"LR: {current_lr:.2e}")

        if val_metrics['RMSE'] < best_rmse:
            best_rmse = val_metrics['RMSE']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_rmse': best_rmse,
                'config': config
            }, output_dir / 'checkpoints' / 'best_model.pth')
            print(f"✓ 保存最佳模型 (epoch {best_epoch}, RMSE: {best_rmse:.4f}m)")
        
        if epoch % config['save_freq'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'config': config
            }, output_dir / 'checkpoints' / f'epoch_{epoch}.pth')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_metrics': val_metrics,
            'config': config
        }, output_dir / 'checkpoints' / 'last.pth')
    
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    plot_training_history(output_dir / 'history.json', output_dir / 'training_curves.png')
    
    print("\n" + "="*80)
    print(f"训练完成！融合stages: {config['fusion_stages']}")
    print("="*80)
    print(f"最佳 RMSE: {best_rmse:.4f}m (Epoch {best_epoch})")
    print(f"实验名称: {config['experiment_id']}")
    print(f"最佳模型: {output_dir / 'checkpoints' / 'best_model.pth'}")
    print(f"训练历史: {output_dir / 'history.json'}")
    print("="*80)


if __name__ == "__main__":
    main()
