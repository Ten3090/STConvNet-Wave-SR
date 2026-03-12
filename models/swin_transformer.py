"""
Swin Transformer Encoder
用于海洋数据超分辨率的Swin Transformer编码器实现

参考论文:
Liu et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" ICCV 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List


class PatchEmbed(nn.Module):
    """图像到补丁嵌入的转换
    
    Args:
        patch_size: 补丁大小
        in_chans: 输入通道数
        embed_dim: 嵌入维度
    """
    
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(in_chans, embed_dim, 
                             kernel_size=patch_size, 
                             stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            x: (B, H', W', C') where H'=H/patch_size, W'=W/patch_size
        """
        x = self.proj(x)  # (B, embed_dim, H', W')
        x = x.permute(0, 2, 3, 1)  # (B, H', W', embed_dim)
        x = self.norm(x)
        return x


class PatchMerging(nn.Module):
    """补丁合并层，用于下采样
    
    将2x2的补丁合并为1个补丁，同时增加通道数
    """
    
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        
    def forward(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            x: (B, H/2, W/2, 2*C)
        """
        B, H, W, C = x.shape
        
        # 确保H和W是偶数
        if H % 2 != 0 or W % 2 != 0:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
            _, H, W, _ = x.shape
        
        # 分成4个子区域
        x0 = x[:, 0::2, 0::2, :]  # (B, H/2, W/2, C)
        x1 = x[:, 1::2, 0::2, :]  # (B, H/2, W/2, C)
        x2 = x[:, 0::2, 1::2, :]  # (B, H/2, W/2, C)
        x3 = x[:, 1::2, 1::2, :]  # (B, H/2, W/2, C)
        
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4*C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2, W/2, 2*C)
        
        return x


def window_partition(x, window_size):
    """将特征图划分为窗口
    
    Args:
        x: (B, H, W, C)
        window_size: 窗口大小
    Returns:
        windows: (B*num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, 
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """将窗口还原为特征图
    
    Args:
        windows: (B*num_windows, window_size, window_size, C)
        window_size: 窗口大小
        H: 原始高度
        W: 原始宽度
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, 
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """窗口内的多头自注意力机制
    
    Args:
        dim: 输入通道数
        window_size: 窗口大小
        num_heads: 注意力头数
        qkv_bias: 是否使用bias
    """
    
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        # 定义相对位置偏置参数
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        
        # 获取每个token的相对位置索引
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
    def forward(self, x, mask=None):
        """
        Args:
            x: (B*num_windows, N, C) where N = window_size * window_size
            mask: (num_windows, N, N) or None
        Returns:
            x: (B*num_windows, N, C)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        
        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size * self.window_size, 
               self.window_size * self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer块
    
    Args:
        dim: 输入通道数
        num_heads: 注意力头数
        window_size: 窗口大小
        shift_size: 移位大小
        mlp_ratio: MLP隐藏层维度比例
    """
    
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            x: (B, H, W, C)
        """
        B, H, W, C = x.shape
        
        shortcut = x
        x = self.norm1(x)
        
        # 填充以确保可以整除window_size
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape
        
        # 循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), 
                                  dims=(1, 2))
            attn_mask = None  # 简化版本，实际使用中需要生成mask
        else:
            shifted_x = x
            attn_mask = None
        
        # 窗口划分
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)
        
        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)
        
        # 反向循环移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), 
                          dims=(1, 2))
        else:
            x = shifted_x
        
        # 去除填充
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        
        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        
        return x


class SwinTransformer(nn.Module):
    """Swin Transformer编码器
    
    用于ERA5海洋数据的特征提取
    
    Args:
        img_size: 输入图像大小
        patch_size: 补丁大小
        in_chans: 输入通道数
        embed_dim: 嵌入维度
        depths: 每个阶段的块数
        num_heads: 每个阶段的注意力头数
        window_size: 窗口大小
    """
    
    def __init__(self, img_size=224, patch_size=4, in_chans=3,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, drop_rate=0.):
        super().__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        
        # 补丁嵌入层
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        
        # 构建各个阶段
        self.layers = nn.ModuleList()
        self.patch_merging_layers = nn.ModuleList()
        
        for i_layer in range(self.num_layers):
            # 当前阶段的维度
            dim = int(embed_dim * 2 ** i_layer)
            
            # Swin Transformer块
            layer = nn.ModuleList([
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate
                )
                for i in range(depths[i_layer])
            ])
            self.layers.append(layer)
            
            # 补丁合并层（除了最后一层）
            if i_layer < self.num_layers - 1:
                self.patch_merging_layers.append(PatchMerging(dim))
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        """前向传播
        
        Args:
            x: (B, C, H, W)
        Returns:
            features: List of (B, H_i, W_i, C_i) for each stage
        """
        # 补丁嵌入
        x = self.patch_embed(x)  # (B, H/4, W/4, embed_dim)
        
        features = []
        
        # 逐层处理
        for i_layer in range(self.num_layers):
            # Swin Transformer块
            for blk in self.layers[i_layer]:
                x = blk(x)
            
            features.append(x)
            
            # 补丁合并（下采样）
            if i_layer < self.num_layers - 1:
                x = self.patch_merging_layers[i_layer](x)
        
        return features



































