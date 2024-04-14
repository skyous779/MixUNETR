'''
Author: skyous 1019364238@qq.com
Date: 2023-12-24 21:51:43
LastEditors: skyous 1019364238@qq.com
LastEditTime: 2024-01-07 10:45:43
FilePath: /prostate158-main/train.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''
import monai
import argparse

from prostate158.utils import load_config
from prostate158.train import SegmentationTrainer
from prostate158.report import ReportGenerator

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')



# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

parser = argparse.ArgumentParser(description='Train a segmentation model.')
parser.add_argument('--config',
                    type=str,
                    required=True,
                    help='path to the config file')
args = parser.parse_args()
config_fn = args.config


config = load_config(config_fn)



monai.utils.set_determinism(seed=config.seed)


import wandb
wandb.init(
                project=config.project if config.project!='' else "prostate158",
                group=config.network,
                config=config,
                # job_type=f"fold_{args.fold}",
            )

print(
    f"""
    Running supervised segmentation training
    Run ID:     {config.run_id}
    Debug:      {config.debug}
    Out dir:    {config.out_dir}
    model dir:  {config.model_dir}
    log dir:    {config.log_dir}
    images:     {config.data.image_cols}
    labels:     {config.data.label_cols}
    data_dir    {config.data.data_dir}
    """
)

# create supervised trainer for segmentation task
trainer=SegmentationTrainer(
    progress_bar=True, 
    early_stopping = True, 
    metrics = ["MeanDice", "HausdorffDistance", "SurfaceDistance"],
    save_latest_metrics = True,
    config=config
)

## add lr scheduler to trainer
trainer.fit_one_cycle()

## let's train
trainer.run()

## finish script with final evaluation of the best model
trainer.evaluate()

## generate a markdown document with segmentation results
report_generator=ReportGenerator(
    config.run_id, 
    config.out_dir, 
    config.log_dir
)

report_generator.generate_report()
