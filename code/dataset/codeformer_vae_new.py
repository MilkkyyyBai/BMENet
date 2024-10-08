from typing import Sequence, Dict, Union, List, Mapping, Any, Optional
import math
import time
import io
import random
import os
from random import *
import nibabel as nib
from torchvision.transforms.functional import to_tensor, to_pil_image
import torch
import SimpleITK as sitk

import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data
from utils.common import instantiate_from_config


class CodeformerDataset(data.Dataset):
    
    def __init__(
        self,
        gt_file: str,
        lq_file:str,
        # file_backend_cfg: Mapping[str, Any],
        # out_size: int,
        crop_type: str,
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        downsample_range: Sequence[float],
        noise_range: Sequence[float],
        jpeg_range: Sequence[int]
    ) -> "CodeformerDataset":
        super(CodeformerDataset, self).__init__()

        self.crop_type = crop_type
        assert self.crop_type in ["none", "center", "random"]
        # degradation configurations
        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        self.downsample_range = downsample_range
        self.noise_range = noise_range
        self.jpeg_range = jpeg_range
        # img1.png
        self.gt_file = gt_file
        self.lq_file = lq_file
 
        gt_imgs = os.listdir(gt_file)
        self.total_imgs = sorted(gt_imgs)
        # self._init_map()

    def load_gt_image(self, image_path: str, max_retry: int = 5) -> Optional[np.ndarray]:
        img = nib.load(image_path).get_fdata()

        
        return  img


# total_imgs:拍过顺序的原本图像  all_imgs：lq image
    def _init_map(self):
        self.degrade_map = {}
        for img_path in self.total_imgs:
            self.degrade_map[img_path] = []
        degrade_imgs = os.listdir(self.lq_file)
        total_imgs = sorted(degrade_imgs)
        for img_path in total_imgs:
            _, img_name = os.path.split(img_path)
            image_id_split = img_name.split('_')[:-1]
            image_id = ''
            for idx, word in enumerate(image_id_split):
                image_id += str(word)
                if idx != len(image_id_split) - 1:
                    image_id += '_'
            image_id += '.nii.gz'
            self.degrade_map[image_id].append(img_path)


    def __getitem__(self, index: int):
        gt_name = self.total_imgs[index % len(self.total_imgs)]
        gt_path = os.path.join(self.gt_file, gt_name)  # 路径
        img_gt = self.load_gt_image(gt_path)

        gt = img_gt.astype(np.float32) # 假设标准化到[0, 1]
        gt = gt[:, :, :, None]  # 从(H, W)转为(H, W, 1)形状


        return gt

    def __len__(self) -> int:
        return len(self.total_imgs)
