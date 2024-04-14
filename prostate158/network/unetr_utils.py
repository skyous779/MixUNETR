'''
Author: skyous 1019364238@qq.com
Date: 2024-01-25 21:08:32
LastEditors: skyous 1019364238@qq.com
LastEditTime: 2024-01-25 21:09:06
FilePath: /prostate158-main/prostate158/network/unetr_utils.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
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
from torch.cuda.amp import autocast

from itertools import repeat
import collections.abc

from monai.networks.nets.basic_unet import Down


