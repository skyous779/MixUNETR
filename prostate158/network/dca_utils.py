import numpy as np
import torch
import torch.nn as nn
import einops

##############DCA###############

class PoolEmbedding(nn.Module):
    def __init__(self,
                pooling,
                patch,
                ) -> None:
        super().__init__()
        self.projection = pooling(output_size=(patch, patch, patch))

    def forward(self, x):
        x = self.projection(x)
        x = einops.rearrange(x, 'B C H W D-> B (H W D) C')        
        return x


class depthwise_conv_block(nn.Module):
    def __init__(self, 
                in_features, 
                out_features,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1), 
                padding=(1, 1, 1), 
                dilation=(1, 1, 1),
                groups=None, 
                norm_type='bn',
                activation=True, 
                use_bias=True,
                pointwise=False, 
                ):
        super().__init__()
        self.pointwise = pointwise
        self.norm = norm_type
        self.act = activation
        self.depthwise = nn.Conv3d(
            in_channels=in_features,
            out_channels=in_features if pointwise else out_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation, 
            bias=use_bias)
        if pointwise:
            self.pointwise = nn.Conv3d(in_features, 
                                        out_features, 
                                        kernel_size=(1, 1, 1), 
                                        stride=(1, 1, 1), 
                                        padding=(0, 0, 0),
                                        dilation=(1, 1, 1), 
                                        bias=use_bias)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm3d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.depthwise(x)
        if self.pointwise:
            x = self.pointwise(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x
    
class depthwise_projection(nn.Module):
    def __init__(self, 
                in_features, 
                out_features, 
                groups,
                kernel_size=(1, 1, 1), 
                padding=(0, 0, 0), 
                norm_type=None, 
                activation=False, 
                pointwise=False) -> None:
        super().__init__()

        self.proj = depthwise_conv_block(in_features=in_features, 
                                        out_features=out_features, 
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        groups=groups,
                                        pointwise=pointwise, 
                                        norm_type=norm_type,
                                        activation=activation)
                            
    def forward(self, x):
        P = int(np.cbrt(x.shape[1]))
        x = einops.rearrange(x, 'B (H W D) C-> B C H W D', H=P, W=P, D=P) 
        # print(x.shape)
        x = self.proj(x)
        # x = einops.rearrange(x, 'B C H W -> B (H W) C')      
        return x
    
class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3, 3),
                 stride=(1, 1, 1),
                 padding=(1, 1, 1),
                 dilation=(1, 1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True, 
                 ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm3d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x

class UpsampleConv(nn.Module):
    def __init__(self, 
                in_features, 
                out_features,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1), 
                norm_type=None, 
                activation=False,
                scale=(2, 2, 2), 
                conv='conv') -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale, 
                              mode='trilinear', 
                              align_corners=True)
        if conv == 'conv':
            self.conv = conv_block(in_features=in_features, 
                                    out_features=out_features, 
                                    kernel_size=(1, 1, 1),
                                    padding=(0, 0, 0),
                                    norm_type=norm_type, 
                                    activation=activation)
        elif conv == 'depthwise':
            self.conv = depthwise_conv_block(in_features=in_features, 
                                    out_features=out_features, 
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    norm_type=norm_type, 
                                    activation=activation)
    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x

