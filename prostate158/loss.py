'''
Author: skyous 1019364238@qq.com
Date: 2023-12-24 21:51:43
LastEditors: skyous 1019364238@qq.com
LastEditTime: 2024-01-04 17:35:54
FilePath: /prostate158-main/prostate158/loss.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''


import monai
from .utils import load_config




def get_loss(config: dict): 
    """Create a loss function of `type` with specific keyword arguments from config.
    Example: 
        
        config.loss
        >>> {'DiceCELoss': {'include_background': False, 'softmax': True, 'to_onehot_y': True}}

        get_loss(config)
        >>> DiceCELoss(
        >>>   (dice): DiceLoss()
        >>>   (cross_entropy): CrossEntropyLoss()
        >>> )
    
    """
    loss_type = list(config.loss.keys())[0]
    loss_config = config.loss[loss_type]
    loss_fun =  getattr(monai.losses, loss_type)
    loss = loss_fun(**loss_config)
    return loss