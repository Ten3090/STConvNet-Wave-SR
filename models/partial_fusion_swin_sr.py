"""
Swin Transformer based fusion model for super-resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .swin_transformer import SwinTransformer
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    from swin_transformer import SwinTransformer


class PartialFusionSwinSR(nn.Module):
    """Swin Transformer based fusion model."""
    
    def __init__(
        self,
        in_channels=4,
        out_channels=1,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=2,
        mlp_ratio=4.,
        drop_rate=0.,
        upscale=5,
        fusion_stages=[1, 2, 3, 4]
    ):
        super().__init__()
        
        self.upscale = upscale
        self.fusion_stages = fusion_stages
        
        self.conv_first = nn.Conv2d(in_channels, embed_dim, 3, 1, 1, bias=False)
        
        self.swin = SwinTransformer(
            img_size=8,
            patch_size=1,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate
        )
        
        stage_dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        
        self.stage_projections = nn.ModuleDict()
        for stage_idx in fusion_stages:
            stage_dim = stage_dims[stage_idx - 1]
            self.stage_projections[f'stage_{stage_idx}'] = nn.Conv2d(
                stage_dim, embed_dim, 1, 1, 0, bias=False
            )
        
        total_channels = embed_dim * len(fusion_stages)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_channels, embed_dim, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv_after_fusion = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * (upscale ** 2), 3, 1, 1, bias=False),
            nn.PixelShuffle(upscale),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.conv_last = nn.Conv2d(embed_dim, out_channels, 3, 1, 1, bias=False)
        
        self.global_residual = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
    
    def forward(self, x):
        """
        Args:
            x: [B, in_channels, 8, 8] low-resolution input
        
        Returns:
            out: [B, out_channels, 40, 40] super-resolved output
        """
        x_input = x
        
        x = self.conv_first(x)
        x_shallow = x
        
        features = self.swin(x)
        
        fused_features = []
        
        for stage_idx in self.fusion_stages:
            feat = features[stage_idx - 1]
            B, H, W, C = feat.shape
            
            feat = feat.permute(0, 3, 1, 2).contiguous()
            
            if H != 8 or W != 8:
                feat = F.interpolate(feat, size=(8, 8), mode='bilinear', align_corners=False)
            
            feat = self.stage_projections[f'stage_{stage_idx}'](feat)
            
            fused_features.append(feat)
        
        fused = torch.cat(fused_features, dim=1)
        fused = self.fusion_conv(fused)
        
        x = fused + x_shallow
        
        x = self.conv_after_fusion(x)
        
        x = self.conv_before_upsample(x)
        x = self.upsample(x)
        
        out = self.conv_last(x)
        
        x_bicubic = F.interpolate(x_input, scale_factor=self.upscale, 
                                   mode='bicubic', align_corners=True)
        out = out + self.global_residual(x_bicubic)
        
        return out


def create_partial_fusion_swin_sr(fusion_stages=[1, 2, 3, 4], in_channels=4):
    model = PartialFusionSwinSR(
        in_channels=in_channels,
        out_channels=1,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=2,
        upscale=5,
        fusion_stages=fusion_stages
    )
    return model
