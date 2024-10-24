'''
Author: skyous 1019364238@qq.com
Date: 2023-12-24 21:51:43
LastEditors: skyous 1019364238@qq.com
LastEditTime: 2024-08-17 17:13:37
FilePath: /prostate158-main/prostate158/model.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''
# create a standard UNet

from monai.networks.nets import UNet
from prostate158.network.unetr import UNETR


from prostate158.network.nnFormer.nnFormer_seg import nnFormer
from prostate158.network.UXNet_3D.network_backbone import UXNET

from prostate158.network.biformer.biformer_unetr import BiformerUNETR

from monai.networks.nets import SwinUNETR
from prostate158.network.mixformer.mixing_unetr import MixingUNETR
from prostate158.network.mixformer.mixing_unetr_k import MixingUNETR as MixingUNETR_K
from prostate158.network.mixformer.mixing_unetr_q import MixingUNETR as MixingUNETR_Q
from prostate158.network.mixformer.mixing_unetr_qk import MixingUNETR as MixingUNETR_QK
from prostate158.network.mixformer.mixing_unetr_qkv import MixingUNETR as MixingUNETR_QKV
from prostate158.network.mixformer.mixing_unetr_non import MixingUNETR as MixingUNETR_NON
# from prostate158.network.mixformer.mixing_unetr_v2 import MixingUNETR as MixingUNETR_V2
# from prostate158.network.mixformer.mixing_unetr_v3 import MixingUNETR as MixingUNETR_V3
# from prostate158.network.mixformer.mixing_unetr_v4 import MixingUNETR as MixingUNETR_V4

def get_model(config: dict):

    # elif config.network == "davitunetr_scconv":
    #     return davitUNETR_scconv(
    #             in_channels=len(config.data.image_cols),
    #             out_channels=config.model.out_channels,
    #             img_size=(96,96,96),
    #             feature_size=24,
    #             depths=config.model.depth,
    #         )
    
    if config.network == "UXNET":
        return UXNET(
                    in_chans=len(config.data.image_cols),
                    out_chans=config.model.out_channels,
                    depths=[2, 2, 2, 2],
                    feat_size=[48, 96, 192, 384],
                    drop_path_rate=0,
                    layer_scale_init_value=1e-6,
                    spatial_dims=3,
                )



    elif config.network == "nnformer":
        return nnFormer(
            input_channels=len(config.data.image_cols), 
            num_classes=config.model.out_channels
            )


    elif config.network == "swinunetr":
        return SwinUNETR(
                in_channels=len(config.data.image_cols),
                out_channels=config.model.out_channels,
                depths=config.model.depth,
                img_size=(96,96,96),
                feature_size=24,
            )
    
    elif config.network == "swinunetr_48":
        return SwinUNETR(
                in_channels=len(config.data.image_cols),
                out_channels=config.model.out_channels,
                depths=config.model.depth,
                img_size=(96,96,96),
                feature_size=48,
            )

    # mixunetr系列
    elif config.network == "mixunetr":
        return MixingUNETR(
                in_channels=len(config.data.image_cols),
                out_channels=config.model.out_channels,
                depths=config.model.depth,
                img_size=(96,96,96),
                feature_size=48,
            )
    elif config.network == "mixunetr_k":    
        return MixingUNETR_K(
                in_channels=len(config.data.image_cols),
                out_channels=config.model.out_channels,
                depths=config.model.depth,
                img_size=(96,96,96),
                feature_size=48,
            )
    elif config.network == "mixunetr_q":
        return MixingUNETR_Q(
                in_channels=len(config.data.image_cols),
                out_channels=config.model.out_channels,
                depths=config.model.depth,
                img_size=(96,96,96),
                feature_size=48,
            )
    elif config.network == "mixunetr_qk":
        return MixingUNETR_QK(
                in_channels=len(config.data.image_cols),
                out_channels=config.model.out_channels,
                depths=config.model.depth,
                img_size=(96,96,96),
                feature_size=48,
            )
    elif config.network == "mixunetr_qkv":
        return MixingUNETR_QKV(
                in_channels=len(config.data.image_cols),
                out_channels=config.model.out_channels,
                depths=config.model.depth,
                img_size=(96,96,96),
                feature_size=48 if config.model.feature_size is None else config.model.feature_size,
                act_layer="GELU" if config.model.act_layer is None else config.model.act_layer,
                # init=config.init,
            )
    elif config.network == "mixunetr_non":
        return MixingUNETR_QKV(
                in_channels=len(config.data.image_cols),
                out_channels=config.model.out_channels,
                depths=config.model.depth,
                img_size=(96,96,96),
                feature_size=48,
            )


    elif config.network == "biformerunetr":
        return BiformerUNETR(
                in_channels=len(config.data.image_cols),
                out_channels=config.model.out_channels,
                depths=config.model.depth,
                img_size=(96,96,96),
                feature_size=48,
            )
    

    elif config.network == "3dunet":
        return UNet(
            spatial_dims=config.ndim,
            in_channels=len(config.data.image_cols),
            out_channels=config.model.out_channels,
            channels=config.model.channels,
            strides=config.model.strides,
            num_res_units=config.model.num_res_units,
            act=config.model.act,
            norm=config.model.norm,
            dropout=config.model.dropout,
                )

    elif config.network == "unetr":
        return UNETR(
                in_channels=len(config.data.image_cols),
                out_channels=config.model.out_channels,
                img_size=(96,96,96),
                feature_size=48,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                pos_embed="perceptron",
                norm_name="instance",
                conv_block=True,
                res_block=True,
                dropout_rate=0.0,
            )
    
    else :
        raise ValueError(f"Unknown network {config.network}")





if __name__ == "__main__":
    # model = UNETR(
    #         in_channels=1,
    #         out_channels=3,
    #         img_size=(96,96,96),
    #         feature_size=32,
    #         hidden_size=768,
    #         mlp_dim=3072,
    #         num_heads=12,
    #         pos_embed="perceptron",
    #         norm_name="instance",
    #         conv_block=True,
    #         res_block=True,
    #         dropout_rate=0.0,
    #     ).to("cpu")
    model = UNet(spatial_dims=3,
                 in_channels=1,
                 out_channels=3,
                 channels=[16, 32, 64, 128, 256, 512],
                 strides=[2, 2, 2, 2, 2],
                 num_res_units=4,
                #  act=PRELU,
                #  norm=BATCH,
                #  dropout=0.15,
                 )
    
    # model = SwinUNETR(
    #     in_channels=1,
    #         out_channels=3,
    #         img_size=(96,96,96),
    #         feature_size=24,
    # ).to("cpu")
    # print(model)
    # from torchkeras import summary  

    # # total = sum([param.nelement() for param in model.parameters()])
    # # print("parameter:%fM" % (total/1e6)) 
    # # summary(model, input_shape=(1, 96, 96, 96))


    # from thop import profile
    # import torch
    # input = torch.randn(1, 1, 96, 96, 96)

    # flops, params = profile(model, inputs=(input,))
    # print('FLOPs = ' + str(flops/1000**3) + 'G')
    # print('Params = ' + str(params/1000**2) + 'M')