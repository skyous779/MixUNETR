import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def tanh_range(l=0.5, r=2.0):
    def get_activation(left, right):
        def activation(x):
            return (torch.tanh(x) * 0.5 + 0.5) * (right - left) + left
        return activation
    return get_activation(l, r)


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(self, in_channels, out_channels, ksize=3, stride=1, bias=True):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv3d(in_channels,
                              out_channels,
                              kernel_size=ksize,
                              stride=stride,
                              padding=pad,
                              bias=bias)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))


class AdaptiveGamma(nn.Module):
    def __init__(self, in_ch=3, nf=32, gamma_range=None):
        super().__init__()

        if gamma_range == None:
            self.gamma_range = [1., 10.]
        else:
            self.gamma_range = gamma_range
        print(f'gamma range: {gamma_range}')

        self.head1 = BaseConv(in_ch, nf, ksize=3, stride=2)
        self.body1 = BaseConv(nf, nf*2, ksize=3, stride=2)
        self.body2 = BaseConv(nf*2, nf*4, ksize=3, stride=2)
        self.body3 = BaseConv(nf*4, nf*2, ksize=3)
        self.pooling = nn.AdaptiveAvgPool3d(1)

        self.image_adaptive_gamma = nn.Sequential(
            nn.Linear(nf*2, nf*4),
            nn.LeakyReLU(inplace=True, negative_slope=0.1),
            nn.Linear(nf*4, in_ch, bias=False)
        )

        # self.out_layer = nn.Sequential(
        #     nn.Conv3d(in_ch, 16, 1, stride=1, padding=0),
        #     nn.LeakyReLU(inplace=True, negative_slope=0.1),
        #     nn.Conv3d(16, in_ch, 1, stride=1, padding=0, bias=False)
        # )

    def apply_gamma(self, img, params):
        params = tanh_range(self.gamma_range[0], self.gamma_range[1])(params)[..., None, None, None]
        # out_image = img ** (1.0 / params)
        out_image = img ** params
        return out_image

    def forward(self, img):
        # img_down = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=True)

        fea = self.head1(img)
        fea_s2 = self.body1(fea)
        fea_s4 = self.body2(fea_s2)
        fea_s8 = self.body3(fea_s4)

        fea_gamma = self.pooling(fea_s8)
        fea_gamma = fea_gamma.view(fea_gamma.shape[0], fea_gamma.shape[1])
        para_gamma = self.image_adaptive_gamma(fea_gamma)
        out_gamma = self.apply_gamma(img, para_gamma)

        # out = self.out_layer(out_gamma)

        return out_gamma

if __name__=='__main__':
    nf = 16
    gamma_range = [0., 1.]
    # net = AdaptiveModule(in_ch=3, nf=nf, gamma_range=gamma_range)
    net = AdaptiveGamma(in_ch=1, nf=nf, gamma_range=gamma_range)

    input_x = torch.tensor(np.random.random((2, 1, 96, 96, 96)), dtype=torch.float32)
    output = net(input_x)
    print(output.shape)