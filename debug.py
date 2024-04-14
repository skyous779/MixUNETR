import torch
input = torch.randn(1, 1, 96, 96, 96)
print(input.shape)
from torchkeras import summary
from thop import profile
from monai.networks.nets import SwinUNETR

from prostate158.network.nnFormer.nnFormer_seg import nnFormer
from prostate158.network.unetr import UNETR

from prostate158.network.biformer.biformer_unetr import BiformerUNETR

from prostate158.network.mixformer.mixing_unetr import MixingUNETR
from prostate158.network.mixformer.mixing_unetr_k import MixingUNETR as MixingUNETR_K
from prostate158.network.mixformer.mixing_unetr_q import MixingUNETR as MixingUNETR_Q
from prostate158.network.mixformer.mixing_unetr_qk import MixingUNETR as MixingUNETR_QK
from prostate158.network.UXNet_3D.network_backbone import UXNET
from monai.networks.nets import UNet


# swin_model = SwinUNETR(
#         img_size=(96, 96, 96),
#         depths = [2,2,2,2],
#         # num_heads = [2, 4, 8, 16],
#         # num_heads = [6,12,24,48],
#         in_channels=1,
#         out_channels=3,
#         feature_size=48,
#         use_checkpoint=False,
#     ).to(device='cpu')

# # summary(model, input_shape=(1, 96, 96, 96))
# print("swin_model:")
# flops, params = profile(swin_model, inputs=(input,))
# print('FLOPs = ' + str(flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')




# model = MixingUNETR_v2(
#             img_size=(96, 96, 96),
#             depths=[2,2,2,2],
#             # num_heads=(3,6,12,24),
#             in_channels=1,
#             out_channels=3,
#             feature_size=48,
#             use_checkpoint=False,
#             drop_rate=0.,
#         ).to(device='cpu')

# flops, params = profile(model, inputs=(input,))
# print('FLOPs = ' + str(flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')

# model = UXNET(
#                     in_chans=1,
#                     out_chans=3,
#                     depths=[2, 2, 2, 2],
#                     feat_size=[48, 96, 192, 384],
#                     drop_path_rate=0,
#                     layer_scale_init_value=1e-6,
#                     spatial_dims=3,
#                 )

# model = UNETR(
#                 in_channels=1,
#                 out_channels=3,
#                 img_size=(96,96,96),
#                 feature_size=48,
#                 hidden_size=768,
#                 mlp_dim=3072,
#                 num_heads=12,
#                 pos_embed="perceptron",
#                 norm_name="instance",
#                 conv_block=True,
#                 res_block=True,
#                 dropout_rate=0.0,
#             ).to(device='cpu')

# model = nnFormer(
#             input_channels=1, 
#             num_classes=3,
#             )

# model = UNet(
#             spatial_dims=3,
#             in_channels=1,
#             out_channels=3,
#             channels=[16, 32, 64, 128, 256, 512],
#             strides=[2, 2, 2, 2, 2],
#             num_res_units=0,
#             act='PRELU',
#             norm='BATCH',
#             dropout=0.15,
#             )
# flops, params = profile(model, inputs=(input,))
# print('FLOPs = ' + str(flops/1000**3) + 'G')
# print('Params = ' + str(params/1000**2) + 'M')

model = MixingUNETR_QK(
            img_size=(96, 96, 96),
            depths=[2,2,6,6],
            # num_heads=(3,6,12,24),
            in_channels=1,
            out_channels=3,
            feature_size=48,
            use_checkpoint=False,
            drop_rate=0.,
        ).to(device='cpu')

flops, params = profile(model, inputs=(input,))
print('FLOPs = ' + str(flops/1000**3) + 'G')
print('Params = ' + str(params/1000**2) + 'M')