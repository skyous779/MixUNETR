# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 20231018_1047
Finished on 20231018_?
@author: skyous
"""
import itertools
from typing import Optional, Sequence, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import

import math

# from networks.mixformer.isp import AdaptiveGamma
# from isp import AdaptiveGamma
rearrange, _ = optional_import("einops", name="rearrange")

############SG-attn#####

class SG_Attention(nn.Module):
    def __init__(self, dim, window_size=None, mask=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=8, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio=sr_ratio
        if sr_ratio>1:
            if mask:
                self.q = nn.Linear(dim, dim, bias=qkv_bias)
                self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
                self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
                if self.sr_ratio==8:
                    f1, f2, f3 = 14*14*56, 56*56, 28*56
                elif self.sr_ratio==4:
                    f1, f2, f3 = 49*28, 14*28, 7*28
                elif self.sr_ratio==2:
                    f1, f2, f3 = 2*14, 1*14, None
                self.f1 = nn.Linear(f1, 1)
                self.f2 = nn.Linear(f2, 1)
                if f3 is not None:
                    self.f3 = nn.Linear(f3, 1)
            else:
                self.sr = nn.Conv3d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
                self.act = nn.GELU()

                self.q1 = nn.Linear(dim, dim//2, bias=qkv_bias)
                self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
                self.q2 = nn.Linear(dim, dim // 2, bias=qkv_bias)
                self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.lepe_linear = nn.Linear(dim, dim)
        self.lepe_conv = nn.Conv3d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def window_partition(self, x, window_size, H, W, D):
        B, num_heads, N, C = x.shape
        x = x.contiguous().view(B*num_heads, N, C).contiguous().view(B*num_heads, H, W, D, C)
        B, H, W, D, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, D // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C).\
            view(-1, window_size*window_size*window_size, C)
        return windows  #(B*numheads*num_windows, window_size, window_size, C)


    def window_reverse(self, windows, window_size, H, W, D, head):
        Bhead = int(windows.shape[0] / (H * W * D / window_size / window_size / window_size))
        x = windows.view(Bhead, H // window_size, W // window_size, D // window_size,  window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(Bhead, H, W, D, -1).view(Bhead//head, head, H, W, D, -1)\
            .contiguous().permute(0,2,3,4,1,5).contiguous().view(Bhead//head, H, W, D, -1).view(Bhead//head, H*W*D, -1)
        return x #(B, H, W, D, C)

    def forward(self, x, H, W, D, mask):

        B, N, C = x.shape
        lepe = self.lepe_conv(
            self.lepe_linear(x).transpose(1, 2).view(B, C, H, W, D)).view(B, C, -1).transpose(-1, -2) # shape same as x
        

        if self.sr_ratio > 1:
            if mask is None:
                # global
                q1 = self.q1(x).reshape(B, N, self.num_heads//2, C // self.num_heads).permute(0, 2, 1, 3)
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W, D)
                x_1 = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_1 = self.act(self.norm(x_1))
                kv1 = self.kv1(x_1).reshape(B, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                k1, v1 = kv1[0], kv1[1] #B head N C

                attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale #B head Nq Nkv
                attn1 = attn1.softmax(dim=-1)
                attn1 = self.attn_drop(attn1)
                x1 = (attn1 @ v1).transpose(1, 2).reshape(B, N, C//2)

                # global_mask_value = torch.mean(attn1.detach().mean(1), dim=1) # B Nk  #max ?  mean ?
                # global_mask_value = F.interpolate(global_mask_value.view(B,1,H//self.sr_ratio,W//self.sr_ratio, D//self.sr_ratio),
                #                                   (H, W, D), mode='nearest')[:, 0]

                # local
                q2 = self.q2(x).reshape(B, N, self.num_heads // 2, C // self.num_heads).permute(0, 2, 1, 3) #B head N C
                kv2 = self.kv2(x_.reshape(B, C, -1).permute(0, 2, 1)).reshape(B, -1, 2, self.num_heads // 2,
                                                                          C // self.num_heads).permute(2, 0, 3, 1, 4)
                k2, v2 = kv2[0], kv2[1]
                q_window = 7
                window_size= 7
                q2, k2, v2 = self.window_partition(q2, q_window, H, W, D), self.window_partition(k2, window_size, H, W, D), \
                             self.window_partition(v2, window_size, H, W, D)
                attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
                # (B*numheads*num_windows, window_size*window_size, window_size*window_size)
                attn2 = attn2.softmax(dim=-1)
                attn2 = self.attn_drop(attn2)

                x2 = (attn2 @ v2)  # B*numheads*num_windows, window_size*window_size, C   .transpose(1, 2).reshape(B, N, C)
                x2 = self.window_reverse(x2, q_window, H, W, D, self.num_heads // 2)

                # local_mask_value = torch.mean(attn2.detach().view(B, self.num_heads//2, 
                #                                                   H//window_size*W//window_size*D//window_size, 
                #                                                   window_size*window_size*window_size, window_size*window_size*window_size, 
                #                                                   ).mean(1), dim=2)
                # local_mask_value = local_mask_value.view(B, H // window_size, W // window_size, D // window_size, window_size, window_size, window_size)
                # local_mask_value=local_mask_value.permute(0, 1, 4, 2, 5, 3, 6).contiguous().view(B, H, W, D)

                # mask B H W
                x = torch.cat([x1, x2], dim=-1)
                x = self.proj(x+lepe)
                # x = self.proj(x)
                x = self.proj_drop(x)
                # # cal mask
                # mask = local_mask_value+global_mask_value
                # mask_1 = mask.view(B, H * W * D)
                # mask_2 = mask.permute(0, 3, 2, 1).reshape(B, H * W * D)
                # mask = [mask_1, mask_2]
            else:
                q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

                # mask [local_mask global_mask]  local_mask [value index]  value [B, H, W]
                # use mask to fuse
                mask_1, mask_2 = mask
                mask_sort1, mask_sort_index1 = torch.sort(mask_1, dim=1)
                mask_sort2, mask_sort_index2 = torch.sort(mask_2, dim=1)
                if self.sr_ratio == 8:
                    token1, token2, token3 = H * W * D // (14 * 14 * 56), H * W * D // (56 * 56), H * W * D // (28 * 56)
                    token1, token2, token3 = token1 // 4, token2 // 2, token3 // 4
                elif self.sr_ratio == 4:
                    token1, token2, token3 = H * W * D // (49 * 28), H * W * D // (14 * 28), H * W * D // (7 * 28)
                    token1, token2, token3 = token1 // 4, token2 // 2, token3 // 4
                elif self.sr_ratio == 2:
                    token1, token2 = H * W * D // (2 * 14), H * W * D // 14
                    token1, token2 = token1 // 2, token2 // 2
                # if self.sr_ratio == 8:
                #     token1, token2, token3 = H * W * D // (14 * 14), H * W * D // 56, H * W * D // 28
                #     token1, token2, token3 = token1 // 4, token2 // 2, token3 // 4
                # elif self.sr_ratio == 4:
                #     token1, token2, token3 = H * W * D // 49, H * W * D // 14, H * W * D // 7
                #     token1, token2, token3 = token1 // 4, token2 // 2, token3 // 4
                # elif self.sr_ratio == 2:
                #     token1, token2 = H * W * D // 2, H * W * D // 1
                #     token1, token2 = token1 // 2, token2 // 2
                # todo
                if self.sr_ratio==4 or self.sr_ratio==8:
                    p1 = torch.gather(x, 1, mask_sort_index1[:, :H * W * D // 4].unsqueeze(-1).repeat(1, 1, C))  # B, N//4, C
                    p2 = torch.gather(x, 1, mask_sort_index1[:, H * W * D // 4:H * W * D // 4 * 3].unsqueeze(-1).repeat(1, 1, C)) # B, N//2, C
                    p3 = torch.gather(x, 1, mask_sort_index1[:, H * W * D // 4 * 3:].unsqueeze(-1).repeat(1, 1, C)) # B, N//4, C

                    seq1 = torch.cat([self.f1(p1.permute(0, 2, 1).reshape(B, C, token1, -1)).squeeze(-1),
                                      self.f2(p2.permute(0, 2, 1).reshape(B, C, token2, -1)).squeeze(-1),
                                      self.f3(p3.permute(0, 2, 1).reshape(B, C, token3, -1)).squeeze(-1)], dim=-1).permute(0,2,1)  # B N C

                    x_ = x.view(B, H, W, D, C).permute(0, 3, 2, 1, 4).reshape(B, H * W * D, C)
                    p1_ = torch.gather(x_, 1, mask_sort_index2[:, :H * W * D // 4].unsqueeze(-1).repeat(1, 1, C))  # B, N//4, C
                    p2_ = torch.gather(x_, 1, mask_sort_index2[:, H * W * D // 4:H * W * D // 4 * 3].unsqueeze(-1).repeat(1, 1, C))
                    p3_ = torch.gather(x_, 1, mask_sort_index2[:, H * W * D // 4 * 3:].unsqueeze(-1).repeat(1, 1, C))
                    seq2 = torch.cat([self.f1(p1_.permute(0, 2, 1).reshape(B, C, token1, -1)).squeeze(-1),
                                      self.f2(p2_.permute(0, 2, 1).reshape(B, C, token2, -1)).squeeze(-1),
                                      self.f3(p3_.permute(0, 2, 1).reshape(B, C, token3, -1)).squeeze(-1)], dim=-1).permute(0,2,1)  # B N C
                elif self.sr_ratio==2:
                    p1 = torch.gather(x, 1, mask_sort_index1[:, :H * W * D // 2].unsqueeze(-1).repeat(1, 1, C))  # B, N//4, C
                    p2 = torch.gather(x, 1, mask_sort_index1[:, H * W * D // 2:].unsqueeze(-1).repeat(1, 1, C))
                    seq1 = torch.cat([self.f1(p1.permute(0, 2, 1).reshape(B, C, token1, -1)).squeeze(-1),
                                      self.f2(p2.permute(0, 2, 1).reshape(B, C, token2, -1)).squeeze(-1)], dim=-1).permute(0, 2, 1)  # B N C

                    x_ = x.view(B, H, W, D, C).permute(0, 3, 2, 1, 4).reshape(B, H * W * D, C)
                    p1_ = torch.gather(x_, 1, mask_sort_index2[:, :H * W * D // 2].unsqueeze(-1).repeat(1, 1, C))  # B, N//4, C
                    p2_ = torch.gather(x_, 1, mask_sort_index2[:, H * W * D // 2:].unsqueeze(-1).repeat(1, 1, C))
                    seq2 = torch.cat([self.f1(p1_.permute(0, 2, 1).reshape(B, C, token1, -1)).squeeze(-1),
                                      self.f2(p2_.permute(0, 2, 1).reshape(B, C, token2, -1)).squeeze(-1)], dim=-1).permute(0, 2, 1)  # B N C

                kv1 = self.kv1(seq1).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4) # kv B heads N C
                kv2 = self.kv2(seq2).reshape(B, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                kv = torch.cat([kv1, kv2], dim=2)
                k, v = kv[0], kv[1]
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)

                # 这个参数导致参数量巨大
                attn = self.attn_drop(attn)


                x = (attn @ v).transpose(1, 2).reshape(B, N, C)
                x = self.proj(x+lepe)
                # x = self.proj(x)
                x = self.proj_drop(x)
                mask=None

        else:

            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            x = self.proj(x+lepe)
            # x = self.proj(x)

            x = self.proj_drop(x)

            mask=None


            # 没有使用swin,导致计算了起飞？


        return x, mask

####################### global block #######################
class SE(nn.Module):
    """
    Squeeze and excitation block
    """

    def __init__(self,
                 inp,
                 oup,
                 expansion=0.25):
        """
        Args:
            inp: input features dimension.
            oup: output features dimension.
            expansion: expansion ratio.
        """

        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class FeatExtract(nn.Module): # 该模块不改变维度
    """
    Feature extraction block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self, dim, keep_dim=False):
        """
        Args:
            dim: feature size dimension.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE(dim, dim),
            nn.Conv3d(dim, dim, 1, 1, 0, bias=False),
        )
        if not keep_dim:
            self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1) # size减半
        self.keep_dim = keep_dim

    def forward(self, x):
        x = x.contiguous()
        x = x + self.conv(x)
        if not self.keep_dim:
            x = self.pool(x)
        return x
    

def _to_channel_last(x):
    """
    Args:
        x: (B, C, H, W, D)

    Returns:
        x: (B, H, W, D, C)
    """
    return x.permute(0, 2, 3, 4, 1)


def _to_channel_first(x):
    """
    Args:
        x: (B, H, W, D, C)

    Returns:
        x: (B, C, H, W, D)
    """
    return x.permute(0, 4, 1, 2, 3)

class GlobalQueryGen(nn.Module):
    """
    Global query generator based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 image_resolution,
                 window_size,
                 num_heads):
        """
        Args:
            dim: feature size dimension.
            input_resolution: input image resolution.
            window_size: window size.
            num_heads: number of heads.

        For instance, repeating log(56/7) = 3 blocks, with input window dimension 56 and output window dimension 7 at
        down-sampling ratio 2. Please check Fig.5 of GC ViT paper for details.
        """

        super().__init__()
        if input_resolution == image_resolution//2:
            self.to_q_global = nn.Sequential(
                nn.ConstantPad3d(4, 0),
                FeatExtract(dim, keep_dim=False),   
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False),
            )

        elif input_resolution == image_resolution//4:
            self.to_q_global = nn.Sequential(
                nn.ConstantPad3d(2, 0),
                FeatExtract(dim, keep_dim=False),
                FeatExtract(dim, keep_dim=False),
            )

        elif input_resolution == image_resolution//8:

            # if window_size == input_resolution:
            #     self.to_q_global = nn.Sequential(
            #         FeatExtract(dim, keep_dim=True)
            #     )

            # else:
            #     self.to_q_global = nn.Sequential(
            #         FeatExtract(dim, keep_dim=True)
            #     )
            self.to_q_global = nn.Sequential(
                nn.ConstantPad3d(1, 0),
                    FeatExtract(dim, keep_dim=False)
                )

        elif input_resolution == image_resolution//16:
            self.to_q_global = nn.Sequential(
                FeatExtract(dim, keep_dim=True)
            )

        self.resolution = input_resolution
        self.num_heads = num_heads
        self.N = window_size * window_size * window_size
        self.dim_head = torch.div(dim, self.num_heads, rounding_mode='floor')

    def forward(self, x):
        x = _to_channel_last(self.to_q_global(x))
        B = x.shape[0]
        x = x.reshape(B, 1, self.N, self.num_heads, self.dim_head).permute(0, 1, 3, 2, 4)
        return x



def window_reverse2(windows, window_size, D, H, W, C):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    if D < window_size[0]:
        window_size = [D, H, W]
    x = windows.view(
        -1, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2],  C)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous().view(-1, C, D, H, W)
    return x


def window_partition2_(x, window_size):
    """ Split the feature map to windows.
    B, C, H, W --> B * H // win * W // win x win*win x C

    Args:
        x: (B, C, H, W)
        window_size (tuple[int]): window size

    Returns:
        windows: (num_windows*B, window_size * window_size, C)
    """
    B, C, D, H, W = x.shape
    if D < window_size[0]:
        window_size = [D, H, W]
    x = x.view(B, C, D // window_size[0], window_size[0],
                     H // window_size[1], window_size[1],
                   W // window_size[2], window_size[2])
    windows = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().view(
        -1, window_size[0] * window_size[1] * window_size[2], C)
    return windows

__all__ = [
    "GCHyUNETR",
    "window_partition",
    "window_reverse",
    "WindowAttention",
    "SwinTransformerBlock",
    "PatchMerging",
    "PatchMergingV2",
    "MERGING_MODE",
    "BasicLayer",
    "SwinTransformer",
]







class GCHyUNETR(nn.Module):
    """
    Swin UNETR based on: "Hatamizadeh et al.,
    Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images
    <https://arxiv.org/abs/2201.01266>"
    """

    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        init="xavier",
        adaptivegamma = False,
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).

        Examples::

            # for 3D single channel input with size (96,96,96), 4-channel output and feature size of 48.
            >>> net = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=4, feature_size=48)

            # for 3D 4-channel input with size (128,128,128), 3-channel output and (2,4,2,2) layers in each stage.
            >>> net = SwinUNETR(img_size=(128,128,128), in_channels=4, out_channels=3, depths=(2,4,2,2))

            # for 2D single channel input with size (96,96), 2-channel output and gradient checkpointing.
            >>> net = SwinUNETR(img_size=(96,96), in_channels=3, out_channels=2, use_checkpoint=True, spatial_dims=2)

        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims)
        patch_size = ensure_tuple_rep(2, spatial_dims)
        window_size = ensure_tuple_rep(7, spatial_dims)
        self.adaptivegamma = adaptivegamma

        if spatial_dims not in (2, 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size):
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")
        
        # if self.adaptivegamma:
        #     self.AdaptiveGamma = AdaptiveGamma(in_ch=1, nf=16, gamma_range=[0.1, 1.])

        self.normalize = normalize

        self.swinViT = SwinTransformer(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.LayerNorm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=2 * feature_size,
            out_channels=2 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=4 * feature_size,
            out_channels=4 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=16 * feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=16 * feature_size,
            out_channels=8 * feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        elif init == 'trunc_normal':
            self.apply(self._init_weights_trunc_normal)
        else:
            raise ValueError(
                f"{init} is not supported for weight initialization, only support 'xavier' and 'trunc_normal'"
            )

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            # logging.info('=> init weight of Linear from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                # logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)





    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            # logging.info('=> init weight of Linear from xavier uniform')
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                # logging.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
            # print("using xavier for Linear")
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            # print("using xavier for LayerNorm and BatchNorm2d")

    def load_from(self, weights):

        with torch.no_grad():
            self.swinViT.patch_embed.proj.weight.copy_(weights["state_dict"]["module.patch_embed.proj.weight"])
            self.swinViT.patch_embed.proj.bias.copy_(weights["state_dict"]["module.patch_embed.proj.bias"])
            for bname, block in self.swinViT.layers1[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers1")
            self.swinViT.layers1[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.reduction.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.weight"]
            )
            self.swinViT.layers1[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers1.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers2[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers2")
            self.swinViT.layers2[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.reduction.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.weight"]
            )
            self.swinViT.layers2[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers2.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers3[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers3")
            self.swinViT.layers3[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.reduction.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.weight"]
            )
            self.swinViT.layers3[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers3.0.downsample.norm.bias"]
            )
            for bname, block in self.swinViT.layers4[0].blocks.named_children():
                block.load_from(weights, n_block=bname, layer="layers4")
            self.swinViT.layers4[0].downsample.reduction.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.reduction.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.weight.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.weight"]
            )
            self.swinViT.layers4[0].downsample.norm.bias.copy_(
                weights["state_dict"]["module.layers4.0.downsample.norm.bias"]
            )

    def forward(self, x_in):
        # if self.adaptivegamma:
        #     x_in = self.AdaptiveGamma(x_in)
        hidden_states_out = self.swinViT(x_in, self.normalize)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        logits = self.out(out)
        return logits


def window_partition(x, window_size):
    """window partition operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x: input tensor.
        window_size: local window size.
    """
    x_shape = x.size()
    if len(x_shape) == 5:
        b, d, h, w, c = x_shape
        x = x.view(
            b,
            d // window_size[0],
            window_size[0],
            h // window_size[1],
            window_size[1],
            w // window_size[2],
            window_size[2],
            c,
        )
        windows = (
            x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
        )
    elif len(x_shape) == 4:
        b, h, w, c = x.shape
        x = x.view(b, h // window_size[0], window_size[0], w // window_size[1], window_size[1], c)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0] * window_size[1], c)
    return windows


def window_reverse(windows, window_size, dims):
    """window reverse operation based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        windows: windows tensor.
        window_size: local window size.
        dims: dimension values.
    """
    if len(dims) == 4:
        b, d, h, w = dims
        x = windows.view(
            b,
            d // window_size[0],
            h // window_size[1],
            w // window_size[2],
            window_size[0],
            window_size[1],
            window_size[2],
            -1,
        )
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    elif len(dims) == 3:
        b, h, w = dims
        x = windows.view(b, h // window_size[0], w // window_size[1], window_size[0], window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


def window_reverse2(windows, window_size, D, H, W, C):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    if D < window_size[0]:
        window_size = [D, H, W]
    x = windows.view(
        -1, D // window_size[0], H // window_size[1], W // window_size[2], window_size[0], window_size[1], window_size[2],  C)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous().view(-1, C, D, H, W)
    return x


def window_partition2_(x, window_size):
    """ Split the feature map to windows.
    B, C, H, W --> B * H // win * W // win x win*win x C

    Args:
        x: (B, C, H, W)
        window_size (tuple[int]): window size

    Returns:
        windows: (num_windows*B, window_size * window_size, C)
    """
    B, C, D, H, W = x.shape
    if D < window_size[0]:
        window_size = [D, H, W]
    x = x.view(B, C, D // window_size[0], window_size[0],
                     H // window_size[1], window_size[1],
                   W // window_size[2], window_size[2])
    windows = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().view(
        -1, window_size[0] * window_size[1] * window_size[2], C)
    return windows


def get_window_size(x_size, window_size, shift_size=None):
    """Computing window size based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        x_size: input size.
        window_size: local window size.
        shift_size: window shifting size.
    """

    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)


class WindowAttention(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        if len(self.window_size) == 3:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        elif len(self.window_size) == 2:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, D, H, W, mask, q_global):
        b, n, c = x.shape # torch.Size([686, 343, 48])
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)  # type: ignore
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x # torch.Size([686, 343, 48])

class WindowAttentionGlobal(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,

        sr_ratio=None,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        if len(self.window_size) == 3:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        elif len(self.window_size) == 2:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, D, H, W, mask, q_global):
        b, n, c = x.shape
        B = q_global.shape[0]
        q_global = q_global.repeat(1, int(b/B), 1, 1, 1)
        q = q_global.reshape(b, self.num_heads, n, c // self.num_heads)
        kv = self.kv(x).reshape(b, n, 2, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)  # type: ignore
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MixAttention(nn.Module):
    """
    Window based multi-head self attention module with relative position bias based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            qkv_bias: add a learnable bias to query, key, value.
            attn_drop: attention dropout rate.
            proj_drop: dropout rate of output.
        """

        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        mesh_args = torch.meshgrid.__kwdefaults__

        if len(self.window_size) == 3:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                    num_heads,
                )
            )
            coords_d = torch.arange(self.window_size[0])
            coords_h = torch.arange(self.window_size[1])
            coords_w = torch.arange(self.window_size[2])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 2] += self.window_size[2] - 1
            relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
            relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1
        elif len(self.window_size) == 2:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
            )
            coords_h = torch.arange(self.window_size[0])
            coords_w = torch.arange(self.window_size[1])
            if mesh_args is not None:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
            else:
                coords = torch.stack(torch.meshgrid(coords_h, coords_w))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += self.window_size[0] - 1
            relative_coords[:, :, 1] += self.window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv = nn.Linear(dim // 2, dim // 2 * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)


        #####################mixformer############
        self.dwconv_kernel_size = 3
        self.proj_attn = nn.Linear(dim, dim // 2)
        self.proj_attn_norm = nn.LayerNorm(dim // 2)
        self.proj_cnn = nn.Linear(dim, dim)
        self.proj_cnn_norm = nn.LayerNorm(dim)        

        # conv branch
        self.dwconv3x3 = nn.Sequential(
            nn.Conv3d(
                dim, dim,
                kernel_size=self.dwconv_kernel_size,
                padding=self.dwconv_kernel_size // 2,
                groups=dim
            ),
            nn.BatchNorm3d(dim),
            nn.GELU()
        )
        self.channel_interaction = nn.Sequential(
            nn.Conv3d(dim, dim // 8, kernel_size=1),
            nn.BatchNorm3d(dim // 8),
            nn.GELU(),
            nn.Conv3d(dim // 8, dim // 2, kernel_size=1),
        )
        self.attn_norm = nn.LayerNorm(dim // 2)
        self.projection = nn.Conv3d(dim, dim // 2, kernel_size=1)
        self.conv_norm = nn.BatchNorm3d(dim // 2)
        self.spatial_interaction = nn.Sequential(
            nn.Conv3d(dim // 2, dim // 16, kernel_size=1),
            nn.BatchNorm3d(dim // 16),
            nn.GELU(),
            nn.Conv3d(dim // 16, 1, kernel_size=1)
        )
        ##########################finish########################

    def forward(self, x, D, H, W,  mask, q_global):
        # x.shape :torch.Size([686, 343, 24])
        X_proj_attn = self.proj_attn(x) # torch.Size([686, 343, 12])
        x_atten = self.proj_attn_norm(X_proj_attn)   # # torch.Size([686, 343, 12])
        x_cnn = self.proj_cnn_norm(self.proj_cnn(x)) # torch.Size([256, 49, 24])

        x_cnn = window_reverse2(x_cnn, self.window_size, D, H, W, x_cnn.shape[-1]) # B C D H W

        x_cnn = self.dwconv3x3(x_cnn) # torch.Size([2, 48, 49, 49, 49])

        channel_interaction = self.channel_interaction(
            F.adaptive_avg_pool3d(x_cnn, output_size=1)) # torch.Size([2, 24, 1, 1, 1]) 
        
        x_cnn = self.projection(x_cnn) # torch.Size([2, 24, 49, 49, 49])


        b, n, c = x_atten.shape # torch.Size([686, 343, 12])
        qkv = self.qkv(x_atten).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4) # torch.Size([3, 686, 3, 343, 4])
        q, k, v = qkv[0], qkv[1], qkv[2] # torch.Size([686, 3, 343, 4])


        # channel interaction # 
        x_cnn2v = torch.sigmoid(channel_interaction).reshape(
            -1, 1, self.num_heads, 1, c // self.num_heads) # torch.Size([2, 1, 3, 1, 8])
        v = v.reshape(
            x_cnn2v.shape[0], -1, self.num_heads, n, c // self.num_heads) # torch.Size([2, 343, 3, 343, 8])
        v = v * x_cnn2v # torch.Size([2, 343, 3, 343, 8]) interaction
        v = v.reshape(-1, self.num_heads, n, c // self.num_heads) # torch.Size([686, 3, 343, 8])

        q = q * self.scale # torch.Size([686, 3, 343, 8])
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)  # type: ignore
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn).to(v.dtype) #torch.Size([686, 3, 343, 343]) atten是没有正确的维度
        x = (attn @ v).transpose(1, 2).reshape(b, n, c) # torch.Size([686, 343, 24])

        # spatial interaction
        x_spatial = window_reverse2(x, self.window_size, D, H, W, c)
        spatial_interaction = self.spatial_interaction(x_spatial)
        x_cnn = torch.sigmoid(spatial_interaction) * x_cnn
        x_cnn = self.conv_norm(x_cnn) # torch.Size([2, 24, 49, 49, 49])
        # B, C, H, W --> B * H // win * W // win x win*win x C
        x_cnn = window_partition2_(x_cnn, self.window_size)  # torch.Size([686, 343, 24])      

        # concat
        x_atten = self.attn_norm(x)
        x = torch.concat([x_atten, x_cnn], axis=-1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x # torch.Size([686, 343, 48])

class SwinTransformerBlock(nn.Module):
    """
    one whole transformer block !!!
    Swin Transformer block based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Sequence[int],
        shift_size: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: str = "GELU",
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        use_checkpoint: bool = False,
        attention=WindowAttentionGlobal,

        sr_ratio: int = 1,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            num_heads: number of attention heads.
            window_size: local window size.
            shift_size: window shift size.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: stochastic depth rate.
            act_layer: activation layer.
            norm_layer: normalization layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(dim)

        self.attn_class = attention

        self.attn = attention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop, dropout_mode="swin")
        
    def forward_part1(self, x, mask_matrix, q_global):
        x_shape = x.size() # torch.Size([2, 48, 48, 48, 24])
        x = self.norm1(x)
        if len(x_shape) == 5:
            b, d, h, w, c = x.shape # 2， 48， 48，48， 24
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            pad_l = pad_t = pad_d0 = 0
            pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
            pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
            pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
            _, dp, hp, wp, _ = x.shape  # torch.Size([2, 49, 49, 49, 48])
            dims = [b, dp, hp, wp]

        elif len(x_shape) == 4:
            b, h, w, c = x.shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            pad_l = pad_t = 0
            pad_b = (window_size[0] - h % window_size[0]) % window_size[0]
            pad_r = (window_size[1] - w % window_size[1]) % window_size[1]
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, hp, wp, _ = x.shape
            dims = [b, hp, wp]

        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        
        # todo:
        ## SG_Attention不需要进行attn。
        if self.attn_class == SG_Attention:
            x, mask = self.attn(shifted_x.view(b,-1, c), dp, hp, wp,  mask=attn_mask)
            shifted_x = x.view(b, dp, hp, wp, c)

        else:
            x_windows = window_partition(shifted_x, window_size) # torch.Size([2, 49, 49, 49, 24]) -> torch.Size([686, 343, 24])
            attn_windows = self.attn(x_windows, dp, hp, wp,  mask=attn_mask, q_global=q_global)
            attn_windows = attn_windows.view(-1, *(window_size + (c,))) # torch.Size([686, 343, 48])
            shifted_x = window_reverse(attn_windows, window_size, dims)


        if any(i > 0 for i in shift_size):
            if len(x_shape) == 5:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
            elif len(x_shape) == 4:
                x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        if len(x_shape) == 5:
            if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
                x = x[:, :d, :h, :w, :].contiguous()
        elif len(x_shape) == 4:
            if pad_r > 0 or pad_b > 0:
                x = x[:, :h, :w, :].contiguous()

        return x

    def forward_part2(self, x):
        x = self.norm2(x)
        x = self.mlp(x)
        return self.drop_path(x)
        # return self.drop_path(self.mlp(self.norm2(x)))

    def load_from(self, weights, n_block, layer):
        root = f"module.{layer}.0.blocks.{n_block}."
        block_names = [
            "norm1.weight",
            "norm1.bias",
            "attn.relative_position_bias_table",
            "attn.relative_position_index",
            "attn.qkv.weight",
            "attn.qkv.bias",
            "attn.proj.weight",
            "attn.proj.bias",
            "norm2.weight",
            "norm2.bias",
            "mlp.fc1.weight",
            "mlp.fc1.bias",
            "mlp.fc2.weight",
            "mlp.fc2.bias",
        ]
        with torch.no_grad():
            self.norm1.weight.copy_(weights["state_dict"][root + block_names[0]])
            self.norm1.bias.copy_(weights["state_dict"][root + block_names[1]])
            self.attn.relative_position_bias_table.copy_(weights["state_dict"][root + block_names[2]])
            self.attn.relative_position_index.copy_(weights["state_dict"][root + block_names[3]])  # type: ignore
            self.attn.qkv.weight.copy_(weights["state_dict"][root + block_names[4]])
            self.attn.qkv.bias.copy_(weights["state_dict"][root + block_names[5]])
            self.attn.proj.weight.copy_(weights["state_dict"][root + block_names[6]])
            self.attn.proj.bias.copy_(weights["state_dict"][root + block_names[7]])
            self.norm2.weight.copy_(weights["state_dict"][root + block_names[8]])
            self.norm2.bias.copy_(weights["state_dict"][root + block_names[9]])
            self.mlp.linear1.weight.copy_(weights["state_dict"][root + block_names[10]])
            self.mlp.linear1.bias.copy_(weights["state_dict"][root + block_names[11]])
            self.mlp.linear2.weight.copy_(weights["state_dict"][root + block_names[12]])
            self.mlp.linear2.bias.copy_(weights["state_dict"][root + block_names[13]])

    def forward(self, x, mask_matrix, q_global):
        shortcut = x
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.forward_part1, x, mask_matrix)
        else:
            x = self.forward_part1(x, mask_matrix, q_global=q_global)
        x = shortcut + self.drop_path(x)
        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.forward_part2, x)
        else:
            x = x + self.forward_part2(x)
        return x


class PatchMergingV2(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(self, dim: int, norm_layer: Type[LayerNorm] = nn.LayerNorm, spatial_dims: int = 3) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)

    def forward(self, x):

        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x = torch.cat(
                [x[:, i::2, j::2, k::2, :] for i, j, k in itertools.product(range(2), range(2), range(2))], -1
            )

        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x = torch.cat([x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))], -1)

        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchMerging(PatchMergingV2):
    """The `PatchMerging` module previously defined in v0.9.0."""

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 4:
            return super().forward(x)
        if len(x_shape) != 5:
            raise ValueError(f"expecting 5D x, got {x.shape}.")
        b, d, h, w, c = x_shape
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x


MERGING_MODE = {"merging": PatchMerging, "mergingv2": PatchMergingV2}


def compute_mask(dims, window_size, shift_size, device):
    """Computing region masks based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer

     Args:
        dims: dimension values.
        window_size: local window size.
        shift_size: shift size.
        device: device.
    """

    cnt = 0

    if len(dims) == 3:
        d, h, w = dims
        img_mask = torch.zeros((1, d, h, w, 1), device=device)
        for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1

    elif len(dims) == 2:
        h, w = dims
        img_mask = torch.zeros((1, h, w, 1), device=device)
        for h in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
            for w in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
                img_mask[:, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask


class BasicLayer(nn.Module):
    """
    Basic Swin Transformer layer in one stage based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: Sequence[int],
        drop_path: list,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        downsample: Optional[nn.Module] = None,
        use_checkpoint: bool = False,
        input_resolution: int = 48,
        sr_ratio: int = 1,
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            depth: number of layers in each stage.
            num_heads: number of attention heads.
            window_size: local window size.
            drop_path: stochastic depth rate.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            norm_layer: normalization layer.
            downsample: an optional downsampling layer at the end of the layer.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
        """

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    # shift_size=self.no_shift, # if (i % 2 == 0) else self.shift_size,
                    shift_size=self.shift_size if i % 2 == 1 and sr_ratio == 1 else self.no_shift,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                    use_checkpoint=use_checkpoint,
                    attention=SG_Attention if (i % 2 == 0) and (sr_ratio != 1) else WindowAttentionGlobal,
                    sr_ratio=sr_ratio,
                )
                for i in range(depth)
            ]
        )
        self.q_global_gen = GlobalQueryGen(dim=dim,
                                        input_resolution=input_resolution, 
                                        image_resolution=96,
                                        window_size=self.window_size[0] if input_resolution != 6 else 6,
                                        num_heads=num_heads)
        self.downsample = downsample
        if callable(self.downsample):
            self.downsample = downsample(dim=dim, norm_layer=norm_layer, spatial_dims=len(self.window_size))

    def forward(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, c, d, h, w = x_shape
            window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
            q_global = self.q_global_gen(x)
            x = rearrange(x, "b c d h w -> b d h w c")
            dp = int(np.ceil(d / window_size[0])) * window_size[0]
            hp = int(np.ceil(h / window_size[1])) * window_size[1]
            wp = int(np.ceil(w / window_size[2])) * window_size[2]
            attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask, q_global)
            x = x.view(b, d, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, "b d h w c -> b c d h w")

        elif len(x_shape) == 4:
            b, c, h, w = x_shape
            window_size, shift_size = get_window_size((h, w), self.window_size, self.shift_size)
            x = rearrange(x, "b c h w -> b h w c")
            hp = int(np.ceil(h / window_size[0])) * window_size[0]
            wp = int(np.ceil(w / window_size[1])) * window_size[1]
            attn_mask = compute_mask([hp, wp], window_size, shift_size, x.device)
            for blk in self.blocks:
                x = blk(x, attn_mask)
            x = x.view(b, h, w, -1)
            if self.downsample is not None:
                x = self.downsample(x)
            x = rearrange(x, "b h w c -> b c h w")
        return x


class SwinTransformer(nn.Module):
    """
    Swin Transformer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"
    https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        in_chans: int,
        embed_dim: int,
        window_size: Sequence[int],
        patch_size: Sequence[int],
        depths: Sequence[int],
        num_heads: Sequence[int],
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[LayerNorm] = nn.LayerNorm,
        patch_norm: bool = False,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        sr_ratios = [8, 4, 2, 1],
        # sr_ratios = [1, 1, 1, 1],
    ) -> None:
        """
        Args:
            in_chans: dimension of input channels.
            embed_dim: number of linear projection output channels.
            window_size: local window size.
            patch_size: patch size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            mlp_ratio: ratio of mlp hidden dim to embedding dim.
            qkv_bias: add a learnable bias to query, key, value.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            drop_path_rate: stochastic depth rate.
            norm_layer: normalization layer.
            patch_norm: add normalization after patch embedding.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: spatial dimension.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
        """

        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=spatial_dims,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()
        down_sample_mod = look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer,
                downsample=down_sample_mod,
                use_checkpoint=use_checkpoint,
                input_resolution=int(96/2**(i_layer+1)),
                sr_ratio = sr_ratios[i_layer],
            )
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

    def forward(self, x, normalize=True):
        x0 = self.patch_embed(x) # torch.Size([2, 48, 48, 48, 48])
        x0 = self.pos_drop(x0)
        x0_out = self.proj_out(x0, normalize)
        x1 = self.layers1[0](x0.contiguous())
        x1_out = self.proj_out(x1, normalize) # torch.Size([2, 96, 24, 24, 24])
        x2 = self.layers2[0](x1.contiguous())
        x2_out = self.proj_out(x2, normalize) # torch.Size([2, 192, 12, 12, 12])
        x3 = self.layers3[0](x2.contiguous()) 
        x3_out = self.proj_out(x3, normalize) # torch.Size([2, 384, 6, 6, 6])
        x4 = self.layers4[0](x3.contiguous())
        x4_out = self.proj_out(x4, normalize) # torch.Size([2, 768, 3, 3, 3])git 
        return [x0_out, x1_out, x2_out, x3_out, x4_out]

if __name__=='__main__':

    import torch
    input = torch.randn(2, 1, 96, 96, 96)
    print(input.shape)
    from torchkeras import summary
    model = GCHyUNETR(
            img_size=(96, 96, 96),
            depths = [2,2,2,2],
            num_heads = [6,12,24,48],
            in_channels=1,
            out_channels=1,
            feature_size=24,
            use_checkpoint=False,
            adaptivegamma=False,
        ).to(device='cpu')
    # output = model(input)
    # print(output.shape)

    ### torchviz
    # from torchviz import make_dot
    # g = make_dot(output)
    # g.render(filename='./myNetModel', view=False, format='pdf')

    # script_module = torch.jit.trace(model, input, strict=False)
    # torch.jit.save(script_module, "model.pt")


    # total = sum([param.nelement() for param in model.parameters()])
    # print("parameter:%fM" % (total/1e6)) 
    summary(model, input_shape=(1, 96, 96, 96))

