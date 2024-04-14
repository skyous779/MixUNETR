


import logging
import os
import sys
import tempfile
from glob import glob

import nibabel as nib
import numpy as np
import torch

from monai.config import print_config
from monai.data import Dataset, DataLoader, create_test_image_3d, decollate_batch
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    Invertd,
    LoadImaged,
    Orientationd,
    Resized,
    SaveImaged,
    ScaleIntensityd,
    Spacingd,
)
from prostate158.network.mixformer.mixing_unetr import MixingUNETR

from prostate158.data import segmentation_dataloaders
from prostate158.model import get_model
from prostate158.utils import load_config
from prostate158.transforms import get_test_transforms

# config = load_config('config_fn')
# network=get_model(config=config).to(config.device)
# train_loader, val_loader=segmentation_dataloaders(
#             config=config, 
#             train=True, 
#             valid=True, 
#             test=False
#         )

checkpoint = '/home/data1/skyous/prostate158_log/checkpoint/mixunetr_42_0.8259.pt'




def main(tempdir):
    print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    print(f"generating synthetic data to {tempdir} (this may take a while)")



    for i in range(5):
        im, _ = create_test_image_3d(128, 128, 128, num_seg_classes=1, channel_dim=-1)
        n = nib.Nifti1Image(im, np.eye(4))
        nib.save(n, os.path.join(tempdir, f"im{i:d}.nii.gz"))

    images = sorted(glob(os.path.join(tempdir, "*.nii.gz")))
    files = [{"img": img} for img in images]

    # define pre transforms
    pre_transforms = Compose(
        [
            LoadImaged(keys="img"),
            EnsureChannelFirstd(keys="img"),
            Spacingd(keys=("img", "pred"), pixdim=(0.5, 0.5, 0.5), mode=("bilinear", 'nearest')),
            Orientationd(keys="img", axcodes="RAS"),
            ScaleIntensityd(keys="img", minv=0, maxv=1),
        ]
    )
    # define dataset and dataloader
    dataset = Dataset(data=files, transform=pre_transforms)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=4)
    # define post transforms
    post_transforms = Compose(
        [
            Activationsd(keys="pred", sigmoid=True),
            Invertd(
                keys="pred",  # invert the `pred` data field, also support multiple fields
                transform=pre_transforms,
                orig_keys="img",  # get the previously applied pre_transforms information on the `img` data field,
                # then invert `pred` based on this information. we can use same info
                # for multiple fields, also support different orig_keys for different fields
                nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                # to ensure a smooth output, then execute `AsDiscreted` transform
                to_tensor=True,  # convert to PyTorch Tensor after inverting
            ),
            AsDiscreted(keys="pred", threshold=0.5),
            SaveImaged(keys="pred", output_dir="./out", output_postfix="seg", resample=False),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MixingUNETR(
                in_channels=1,
                out_channels=3,
                depths=[2,2,2,2],
                img_size=(96,96,96),
                feature_size=48,
            ).to(device)
    net.load_state_dict(torch.load(checkpoint))

    net.eval()
    with torch.no_grad():
        for d in dataloader:
            images = d["img"].to(device)
            # define sliding window size and batch size for windows inference
            d["pred"] = sliding_window_inference(inputs=images, roi_size=(96, 96, 96), sw_batch_size=4, predictor=net)
            # decollate the batch data into a list of dictionaries, then execute postprocessing transforms
            d = [post_transforms(i) for i in decollate_batch(d)]


if __name__ == "__main__":
    # with tempfile.TemporaryDirectory() as tempdir:
    tempdir = './image'
    main(tempdir)