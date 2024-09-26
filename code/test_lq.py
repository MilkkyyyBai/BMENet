import os
from argparse import ArgumentParser


from torch.nn import functional as F

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from utils.common import instantiate_from_config


import os

from random import choice
from pathlib import Path
from typing import Tuple, Dict, List, Any

import mlflow
import numpy as np
import torch
import torch.nn.functional as F

import math
from argparse import ArgumentParser, Namespace
from pathlib import Path
from time import perf_counter
from typing import Any, Tuple


from models.ddpm_v2_conditioned import DDPM
import numpy as np
import torch
import nibabel as nib


from models.ddpm_v2_conditioned import DDPM
from models.aekl_no_attention import Decoder
from models.aekl_no_attention import AutoencoderKL
from project.utils.utils import (
    sampling_from_ddim,
    sample_from_ddpm
)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from project.utils.cond_fn import *
import torch.nn as nn
import torchvision.models.video as models
from torchvision.models.video import R3D_18_Weights

def init_cond_fn(args) -> None:
    if not args.or_not:
        return None
    if args.g_loss == "mse":
        cond_fn_cls = MSEGuidance
    elif args.g_loss == "w_mse":
        cond_fn_cls = WeightedMSEGuidance
    elif args.g_loss == 'ncc':
        cond_fn_cls = NCCGuidace
    else:
        raise ValueError(args.g_loss)
    cond_fn = cond_fn_cls(
        scale=args.g_scale, t_start=args.g_start, t_stop=args.g_stop,
        space=args.g_space, repeat=args.g_repeat
    )
    return cond_fn
 
def log_txt_as_img(wh, xc):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        # font = ImageFont.truetype('font/DejaVuSans.ttf', size=size)
        font = ImageFont.load_default()
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts

def rgb2ycbcr_pt(img, y_only=False):
    """Convert RGB images to YCbCr images (PyTorch version).

    It implements the ITU-R BT.601 conversion for standard-definition television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    Args:
        img (Tensor): Images with shape (n, 3, h, w), the range [0, 1], float, RGB format.
         y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        (Tensor): converted images with the shape (n, 3/1, h, w), the range [0, 1], float.
    """
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.
    return out_img

def calculate_psnr_pt(img, img2, crop_border, test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) (PyTorch version).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    mse = torch.mean((img - img2)**2, dim=[1, 2, 3])
    return 10. * torch.log10(1. / (mse + 1e-8))

def setup_noise_inputs(
    device: torch.device, 
    img,
    batchsize,
    c_txt,
    # hparams: Namespace
# ) -> Tuple[torch.Tensor, torch.Tensor]:
) -> Tuple[torch.Tensor]:
    
    cond = dict(
            # c_txt = torch.tensor([[gender, age_normalized, ventricular, brain_volume]],device=device),
            c_txt = c_txt,
            c_img = img,  
    )  # shape: [1, 4]

    latent_variable = torch.randn([batchsize, 3, 20, 28, 20], device=device)
    return cond, latent_variable

def get_middle_slice(image):
    """Extract the middle slice along each axis."""
    slices = []
    for axis in range(3):
        mid_index = image.shape[axis] // 2
        slices.append(torch.index_select(image, axis, torch.tensor(mid_index)).squeeze())
    return slices

def cond_3d_pretrain_model():
    # 加载预训练的 3D ResNet-18 模型
    model = models.r3d_18(weights=R3D_18_Weights.DEFAULT)

    # 修改第一层卷积层以适应单通道输入
    # 获取原始的第一层卷积层
    original_conv1 = model.stem[0]

    # 创建一个新的卷积层，具有相同的输出通道数和卷积参数，但输入通道为1
    new_conv1 = nn.Conv3d(
        in_channels=1,
        out_channels=original_conv1.out_channels,
        kernel_size=original_conv1.kernel_size,
        stride=original_conv1.stride,
        padding=original_conv1.padding,
        bias=False
    )

    # 将原始卷积层的权重平均到新的单通道卷积层上
    with torch.no_grad():
        new_conv1.weight = nn.Parameter(original_conv1.weight.mean(dim=1, keepdim=True))

    # 替换模型的第一层卷积层
    model.stem[0] = new_conv1

    # 修改最后的全连接层以适应四分类任务
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 4)

    return model

def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    set_seed(231)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)
    if accelerator.is_local_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")

    # Create model:
    vae = AutoencoderKL(embed_dim=3) # describe the graph shape

    vae_state_dict = torch.load('vae.pth')

    new_state_dict = {}
    for key, value in vae_state_dict.items():
        if key.startswith('vae.'):
            new_key = key[len('vae.'):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    vae.load_state_dict(new_state_dict, strict=True)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    vae = vae.to(device)

    unet_config={
    "params": {
    'image_size' : 64, #notused
    'in_channels' : 7,
    'model_channels' : 256,
    'out_channels' : 3,
    'num_res_blocks' : 2,
    'attention_resolutions' : [8,4,2],
    'dropout' : 0,
    'channel_mult' : (1, 2, 3),
    'conv_resample': False,
    'num_classes': None,
    
    'num_heads':1,
    'num_head_channels':-1,
    'num_heads_upsample':-1,
    'use_scale_shift_norm':False,
    'resblock_updown': True,
    'use_spatial_transformer':True,  # custom transformer support
    'transformer_depth':1,  # custom transformer support
    'context_dim':4,  # custom transformer support
    'n_embed':None
    }
    }

    cldm_config = { 
    "params": 
    {'image_size' : 32,
     'in_channels' : 3, 
     'model_channels' : 256, 
     'hint_channels' : 6, 
     'num_res_blocks' : 2, 
    'attention_resolutions' : [ 8, 4, 2 ],
    'channel_mult' : [ 1, 2, 3 ],
    'num_head_channels' : 1 ,# need to fix for flash-attn
    # 'use_spatial_transformer' : True,
    # 'use_linear_in_transformer' : True,
    'transformer_depth' : 1,
    # 'context_dim' : 4,
    # 'legacy' : False,
    }}

    diffusion = DDPM(unet_config , cldm_config) # describe the graph shape
    diffusion.setVAE(vae)

    sd = torch.load('/data1/baihy/denoise/diffusion.pt', map_location="cpu")
    unused = diffusion.load_pretrained_sd(sd)

    if cfg.train.resume:
        
    #   generatornet.load_state_dict({k.replace('module.',''):v for k,v in torch.load(args.resume).items()})
        controlnet_sd = torch.load(cfg.train.resume, map_location="cpu")
        # init_sd = {}
        # scratch_sd = diffusion.controlnet.state_dict()
        # for key in scratch_sd:
        #     target_key = ".".join(['controlnet.module', key])
        #     init_sd[key] = controlnet_sd[target_key].clone()
        # diffusion.load_controlnet_from_ckpt(controlnet_sd)
        diffusion.load_controlnet_from_ckpt({k.replace('module.',''):v for k,v in controlnet_sd.items()})

    device = accelerator.device
    decoder = torch.load('/data1/baihy/denoise/weight/decoder/data/model.pth')
    decoder = decoder.to(device)
    for p in decoder.parameters():
        p.requires_grad = False

    cond_pretrain = cond_3d_pretrain_model().train().to(device)

    if cfg.train.cond_resume:
        cond_sd = torch.load(cfg.train.cond_resume, map_location="cpu")
        cond_pretrain.load_state_dict({k.replace('module.',''):v for k,v in cond_sd.items()}, strict=True)


    diffusion.to(device)
    cond_fn = init_cond_fn(cfg.test.guidance)

    # Setup optimizer:
    opt = torch.optim.AdamW(diffusion.controlnet.parameters(), lr=cfg.train.learning_rate)
    
    # Setup data:
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset, batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True, drop_last=True
    )
    val_dataset = instantiate_from_config(cfg.dataset.val)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False, drop_last=False
    )

    if accelerator.is_local_main_process:
        print(f"Dataset contains {len(dataset):,} ")

    # Prepare models for training:
    diffusion.eval().to(device)
    decoder.eval().to(device)
    diffusion.controlnet, opt, loader, val_loader = accelerator.prepare(diffusion.controlnet, opt, loader, val_loader)
    # pure_cldm: ControlLDM = accelerator.unwrap_model(diffusion)

    grad_vae = AutoencoderKL(embed_dim=3)
    grad_sd = torch.load(cfg.train.gard_resume, map_location="cpu")
    grad_vae.load_state_dict({k.replace('module.',''):v for k,v in grad_sd.items()}, strict=True)
    grad_vae = grad_vae.eval().to(device)
    for p in grad_vae.parameters():
        p.requires_grad = False
    
   
    global_step = 0
    max_steps = cfg.train.train_steps
  
    # sampler = SpacedSampler(diffusion.betas)
    if accelerator.is_local_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps...")
    

        diffusion.controlnet.eval()
        cond_pretrain.eval()

        for lq, lq_grad, info in val_loader:
            lq = rearrange(lq, "b h w z c -> b c h w z").contiguous().float().to(device)
            lq_grad = rearrange(lq_grad, "b h w z c -> b c h w z").contiguous().float().to(device)
            

            cond_fn.load_target(lq)

            #计算loss promt为空
            with torch.no_grad():  # 训练的时候 这下面的三个 都是相当于测试 全部冻住了（蓝色），只需要向前forward
                
                z_lq = vae.encode(lq)
                z_lq_grad = grad_vae.encode(lq_grad)
                z_hit = torch.cat([z_lq, z_lq_grad], dim=1)
                c_txt = cond_pretrain(lq)
                # c_txt = torch.tensor([[0.5, 0.5, 0.2, 0.5]],device=device)
                
                print(c_txt)
                # if accelerator.is_local_main_process:
                #     print(z_gt.shape)
                cond, latent_variable = setup_noise_inputs(device=device, img = z_hit, batchsize = 1, c_txt = c_txt) 
    
            with torch.no_grad():
                synth_img = sample_from_ddpm(
                    ddpm = diffusion,
                    decoder = vae.reconstruct_ldm_outputs,
                    img=latent_variable,
                    device=accelerator.device,
                    cond=cond,
                    cond_fn=cond_fn,
                    num_timesteps=1000
                )
                    
                
            # 注意，以下转换中假设val_gt和val_lq的shape是[B, H, W, 1]，灰度图1个通道
            # 由于灰度图像只有一个通道，这里将最后一个维度去除了
                for i in range(lq.size(0)):
                    # single_image = val_pred[i].clamp(0, 1).cpu().numpy()
                    single_clean = synth_img[i].squeeze(0).clamp(0, 1).cpu().numpy()
                    print(single_clean.shape)
                    new_filename2 = f"{info['name'][i].split('.')[0]}"
                
                    os.makedirs('results/ADNI/stage2/', exist_ok=True)  # 确保目录存在
                    original_volume = nib.load('0030_0.nii.gz')
                    image_clean_nifti = nib.Nifti1Image(single_clean, affine=original_volume.affine)
                    nib.save(image_clean_nifti, os.path.join('results/ADNI/stage2/', new_filename2))




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    print(args.config)
    main(args)
