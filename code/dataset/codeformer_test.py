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

import numpy as np
import cv2
from PIL import Image
import torch.utils.data as data


from utils.common import instantiate_from_config



class CodeformerDataset(data.Dataset):
    
    def __init__(
        self,
        gt_file: str,
        grad_file: str,
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
        # self.gt_file_list = gt_file_list
        # self.gt_image_files = load_file_list(gt_file_list)
        # self.lq_file_list = lq_file_list
        # self.lq_image_files = load_file_list(lq_file_list)
        # self.file_backend = instantiate_from_config(file_backend_cfg)
        # self.out_size = out_size
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
        self.grad_file = grad_file
        self.lq_file = lq_file
 
        lq_imgs = os.listdir(lq_file)
        self.total_imgs = sorted(lq_imgs)
        # self._init_map()

    def load_lq_image(self, image_path: str, max_retry: int = 5) -> Optional[np.ndarray]:
        img = nib.load(image_path).get_fdata()
     
    
        return img

    

    def __getitem__(self, index: int):
        lq_name = self.total_imgs[index % len(self.total_imgs)]
        lq_path = os.path.join(self.lq_file, lq_name)  # 路径
        # img_lq, shape = self.load_lq_image(lq_path)
        img_lq = self.load_lq_image(lq_path)

        grad_path = os.path.join(self.grad_file, f'grad_{lq_name}')  # 路径
        img_grad = self.load_lq_image(grad_path)


        # 对于灰度图像，需要调整图像的形状，以添加一个单通道的维度
        lq = img_lq.astype(np.float32)
        lq = lq[:, :, :, None]   # 从(H, W)转为(H, W, 1)形状
        grad = img_grad.astype(np.float32) # 假设标准化到[0, 1]
        grad = grad[:, :, :, None]  # 从(H, W)转为(H, W, 1)形状

        prompt = ""
        info = dict()
        
        info['name'] = lq_name
        return lq, grad, info

    def __len__(self) -> int:
        return len(self.total_imgs)

