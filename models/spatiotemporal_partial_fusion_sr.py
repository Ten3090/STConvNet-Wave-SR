"""
Spatiotemporal super-resolution model with ConvLSTM and Swin-based fusion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .partial_fusion_swin_sr import PartialFusionSwinSR
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from partial_fusion_swin_sr import PartialFusionSwinSR


class ConvLSTMCell(nn.Module):
    """ConvLSTM Cell - 保持空间结构的 LSTM"""
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=True
        )
    
    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next


class ConvLSTM(nn.Module):
    """多层 ConvLSTM"""
    def __init__(self, input_dim, hidden_dim, num_layers=2, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList([
            ConvLSTMCell(
                input_dim=input_dim if i == 0 else hidden_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size
            )
            for i in range(num_layers)
        ])
    
    def forward(self, x, return_all_frames=False):
        """
        Args:
            x: [B, T, C, H, W]
        Returns:
            [B, hidden_dim, H, W] 最后时刻特征
        """
        B, T, C, H, W = x.shape
        
        h = [torch.zeros(B, self.hidden_dim, H, W, device=x.device) 
             for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.hidden_dim, H, W, device=x.device) 
             for _ in range(self.num_layers)]
        
        outputs = []
        
        for t in range(T):
            x_t = x[:, t]
            
            for layer in range(self.num_layers):
                h[layer], c[layer] = self.cells[layer](
                    x_t if layer == 0 else h[layer-1],
                    h[layer],
                    c[layer]
                )
            
            outputs.append(h[-1])
        
        if return_all_frames:
            return torch.stack(outputs, dim=1)
        else:
            return outputs[-1]


class SpatioTemporalPartialFusionSR(nn.Module):
    """Spatiotemporal super-resolution model."""
    def __init__(
        self, 
        num_frames=5,
        fusion_stages=[1, 2, 3, 4],
        temporal_dim=64,
        use_temporal=True,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=4,
        upscale=5
    ):
        super().__init__()
        self.num_frames = num_frames
        self.use_temporal = use_temporal
        self.temporal_dim = temporal_dim
        self.fusion_stages = fusion_stages
        self.upscale = upscale
        
        self.shallow_feat = nn.Conv2d(4, temporal_dim, 3, 1, 1)
        
        if use_temporal:
            self.temporal_encoder = ConvLSTM(
                input_dim=temporal_dim,
                hidden_dim=temporal_dim,
                num_layers=2,
                kernel_size=3
            )
        else:
            self.temporal_encoder = None
        
        self.spatial_sr = PartialFusionSwinSR(
            in_channels=temporal_dim,
            out_channels=1,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            upscale=upscale,
            fusion_stages=fusion_stages
        )
    
    def forward(self, x_seq):
        """
        Args:
            x_seq: [B, T, 4, 8, 8]
        Returns:
            [B, 1, 40, 40]
        """
        B, T, C, H, W = x_seq.shape
        
        shallow_feats = []
        for t in range(T):
            feat = self.shallow_feat(x_seq[:, t])
            shallow_feats.append(feat)
        
        shallow_feats = torch.stack(shallow_feats, dim=1)
        
        if self.use_temporal:
            temporal_feat = self.temporal_encoder(shallow_feats)
        else:
            temporal_feat = shallow_feats[:, -1]
        
        hr_output = self.spatial_sr(temporal_feat)
        
        return hr_output


def create_spatiotemporal_partial_fusion_sr(
    num_frames=5, 
    fusion_stages=[1, 2, 3, 4],
    use_temporal=True,
    upscale=5
):
    model = SpatioTemporalPartialFusionSR(
        num_frames=num_frames,
        fusion_stages=fusion_stages,
        temporal_dim=64,
        use_temporal=use_temporal,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=4,
        upscale=upscale
    )
    return model
