import torch
import torch.nn as nn

import numpy as np
from davitUNETR_v3_2_dca import BasicLayer

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


class DCA(nn.Module):
    def __init__(self,
                features,
                strides,
                patch=12,
                n=1, 
                channel_att=True,
                spatial_att=True,   
                ):
        super().__init__()
        self.n = n
        self.features = features
        self.channel_att = channel_att
        self.spatial_att = spatial_att
        self.patch = patch
        self.patch_avg = nn.ModuleList([PoolEmbedding(
                                                    pooling = nn.AdaptiveAvgPool3d,            
                                                    patch=patch, 
                                                    )
                                                    for _ in features])                
        self.avg_map = nn.ModuleList([depthwise_projection(in_features=feature,
                                                            out_features=feature, 
                                                            kernel_size=(1, 1, 1),
                                                            padding=(0, 0, 0), 
                                                            groups=feature
                                                            )
                                                    for feature in features])
        self.attention =  BasicLayer(
                            dim=196,
                            depth=2,
                            num_heads=12,
                            window_size=(7, 7, 7),
                            drop_path=[0.0, 0.0],
                            mlp_ratio=4.0,
                            qkv_bias=True,
                            drop=0.0,
                            attn_drop=0.0,
                            norm_layer=nn.LayerNorm,
                            downsample=None,
                            use_checkpoint=False,
                            qkv_mode='',
                        )  
                                
        # self.attention = nn.ModuleList([
        #                                 CCSABlock(features=features, 
        #                                           channel_head=channel_head, 
        #                                           spatial_head=spatial_head, 
        #                                           channel_att=channel_att, 
        #                                           spatial_att=spatial_att) 
        #                                           for _ in range(n)])
                     
        self.upconvs = nn.ModuleList([UpsampleConv(in_features=feature, 
                                                    out_features=feature,
                                                    kernel_size=(1, 1, 1),
                                                    padding=(0, 0, 0),
                                                    norm_type=None,
                                                    activation=False,
                                                    scale=stride, 
                                                    conv='conv')
                                                    for feature, stride in zip(features, strides)])                                                      
        self.bn_relu = nn.ModuleList([nn.Sequential(
                                                    nn.BatchNorm3d(feature), 
                                                    nn.ReLU()
                                                    ) 
                                                    for feature in features])
    
    def forward(self, raw):

        x = self.m_apply(raw, self.patch_avg) 
        x = self.m_apply(x, self.avg_map) # depth wish conv block, don't change the shape of x



        combined_tensor = torch.cat(x, dim=1)
        print(combined_tensor.shape)

        for block in self.attention:
            x = block(x)

        x = torch.split(x, self.features, dim=1)

        # x = [self.reshape(i) for i in x]
        x = self.m_apply(x, self.upconvs)
        x_out = self.m_sum(x, raw)
        x_out = self.m_apply(x_out, self.bn_relu)
        return (*x_out, )      

    def m_apply(self, x, module):
        return [module[i](j) for i, j in enumerate(x)]

    def m_sum(self, x, y):
        return [xi + xj for xi, xj in zip(x, y)]  
        
    def reshape(self, x):
        return einops.rearrange(x, 'B (H W D) C-> B C H W D', H=self.patch, W=self.patch, D=self.patch) 
    

if __name__ == '__main__':
    # model = DCA(features=(24, 24, 48, 96),
    #             strides=(8, 4, 2, 1),
    #             patch=12,)

    # input = [torch.randn(1, 24, 96, 96, 96),
    #          torch.randn(1, 24, 48, 48, 48),
    #          torch.randn(1, 48, 24, 24, 24),
    #          torch.randn(1, 96, 12, 12, 12)]

    # output = model(input)
    # for i in output:
    #     print(i.shape)

    # 定义输入 tensor
    input_tensor = torch.randn(1, 192, 96, 96, 96)

    # 将输入 tensor 按照 (24, 24, 48, 96) 进行拆分
    split_tensor_list = torch.split(input_tensor, (24, 24, 48, 96), dim=1)


    # 打印拆分后每个 tensor 的 shape
    for i, split_tensor in enumerate(split_tensor_list):
        print(f'Split Tensor {i} shape: {split_tensor.size()}')