'''
Author: skyous 1019364238@qq.com
Date: 2023-12-24 21:51:43
LastEditors: skyous 1019364238@qq.com
LastEditTime: 2024-01-18 11:26:33
FilePath: /prostate158-main/prostate158/utils.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''

import os
import yaml
import torch
import monai
import munch 
import datetime

def load_config(fn: str='config.yaml'):
    "Load config from YAML and return a serialized dictionary object"
    with open(fn, 'r') as stream:
        config=yaml.safe_load(stream)
    config=munch.munchify(config)

    # add time for train
    now = datetime.datetime.now()
    formatted_date_time = now.strftime("%Y%m%d_%H%M_")
    config.run_id += config.project
    config.run_id += formatted_date_time
    config.run_id += config.network


    if not config.overwrite:
        i=1
        while os.path.exists(os.path.join(config.save_path, config.run_id+f'_{i}')):
            i+=1
        config.run_id+=f'_{i}'

    config.out_dir = os.path.join(config.save_path, config.run_id, config.out_dir)
    config.log_dir = os.path.join(config.save_path, config.run_id, config.log_dir)
    
    if not isinstance(config.data.image_cols, (tuple, list)): 
        config.data.image_cols=[config.data.image_cols]
    if not isinstance(config.data.label_cols, (tuple, list)): 
        config.data.label_cols=[config.data.label_cols]
    
    config.transforms.mode=('bilinear', ) * len(config.data.image_cols) + \
                             ('nearest', ) * len(config.data.label_cols)
    return config

    
def num_workers():
    "Get max supported workers -2 for multiprocessing"
    import resource
    import multiprocessing
    
    # first check for max number of open files allowed on system
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    n_workers=multiprocessing.cpu_count() - 2
    # giving each worker at least 256 open processes should allow them to run smoothly
    max_workers = soft_limit // 256
    if max_workers < n_workers: 
        print(
            "Will not use all available workers as number of allowed open files is to small"
            "to ensure smooth multiprocessing. Current limits are:\n" 
            f"\t soft_limit: {soft_limit}\n"
            f"\t hard_limit: {hard_limit}\n" 
            "try increasing the limits to at least {256*n_workers}." 
            "See https://superuser.com/questions/1200539/cannot-increase-open-file-limit-past-4096-ubuntu"
            "for more details"
        )
        return max_workers
              
    return n_workers

USE_AMP=True if monai.utils.get_torch_version_tuple() >= (1, 6) else False