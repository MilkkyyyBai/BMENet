import os
from argparse import ArgumentParser
import warnings
import lpips

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
import random
from argparse import Namespace
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


from models.ddpm_v2_conditioned import DDPM
from models.aekl_no_attention import Decoder
from models.aekl_no_attention import AutoencoderKL, OnlyEncoder
from project.utils.utils import (
    sampling_from_ddim,
    sample_from_ddpm,
)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch.nn as nn
import torchvision.models.video as models
from torchvision.models.video import R3D_18_Weights

 
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
) -> Tuple[torch.Tensor]:

    
    cond = dict(
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

    # with warnings.catch_warnings():
    #     # avoid warnings from lpips internal
    #     warnings.simplefilter("ignore")
    #     lpips_model = lpips.LPIPS(net="alex", verbose=accelerator.is_local_main_process).eval().to(device)
        

    # Setup an experiment folder:
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
    'use_spatial_transformer' : False,
    # 'use_linear_in_transformer' : True,
    'transformer_depth' : 1,
    # 'context_dim' : 4,
    # 'legacy' : False,
    }}

    diffusion = DDPM(unet_config , cldm_config) # describe the graph shape


    sd = torch.load('/data1/baihy/denoise/diffusion.pt', map_location="cpu")
    unused = diffusion.load_pretrained_sd(sd)

    if cfg.train.resume:
        diffusion.load_controlnet_from_ckpt(torch.load(cfg.train.resume, map_location="cpu"))

    device = accelerator.device

    diffusion.to(device)

    grad_vae = AutoencoderKL(embed_dim=3)
    grad_sd = torch.load(cfg.train.gard_resume, map_location="cpu")
    grad_vae.load_state_dict({k.replace('module.',''):v for k,v in grad_sd.items()}, strict=True)
    grad_vae = grad_vae.eval().to(device)
    for p in grad_vae.parameters():
        p.requires_grad = False


    cond_pretrain = cond_3d_pretrain_model().eval().to(device)
    if cfg.train.cond_resume:
        cond_sd = torch.load(cfg.train.cond_resume, map_location="cpu")
        cond_pretrain.load_state_dict({k.replace('module.',''):v for k,v in cond_sd.items()}, strict=True)
    for p in cond_pretrain.parameters():
        p.requires_grad = False


    opt_diff = torch.optim.AdamW(diffusion.controlnet.parameters(), lr=cfg.train.learning_rate)

    
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
    diffusion.controlnet.train().to(device)
    diffusion.controlnet, cond_pretrain, opt_diff, loader, val_loader = accelerator.prepare(diffusion.controlnet, cond_pretrain, opt_diff, loader, val_loader)
    # pure_cldm: ControlLDM = accelerator.unwrap_model(diffusion)
    
    
    # Variables for monitoring/logging purposes:
    global_step = 0
    max_steps = cfg.train.train_steps
    step_loss = []
    epoch = 0
    epoch_loss = []
    # sampler = SpacedSampler(diffusion.betas)
    if accelerator.is_local_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps...")
    
    while global_step < max_steps:
        pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch", total=len(loader))
        for gt, lq, lq_grad in loader:
            gt = rearrange(gt, "b h w z c -> b c h w z").contiguous().float().to(device)
            lq = rearrange(lq, "b h w z c -> b c h w z").contiguous().float().to(device)
            lq_grad = rearrange(lq_grad, "b h w z c -> b c h w z").contiguous().float().to(device)
           

            #计算loss promt为空
            with torch.no_grad():  # 训练的时候 这下面的三个 都是相当于测试 全部冻住了（蓝色），只需要向前forward
                c_txt = cond_pretrain(lq)
                z_gt = vae.encode(gt) # -1，1之间
                z_lq = vae.encode(lq)
                z_lq_grad = grad_vae.encode(lq_grad)
                z_hit = torch.cat([z_lq, z_lq_grad], dim=1)
                # if accelerator.is_local_main_process:
                #     print(z_gt.shape)
                cond, latent_variable = setup_noise_inputs(device=device, img = z_hit, batchsize = 1, c_txt = c_txt) 

            t = torch.randint(0, diffusion.num_timesteps, (z_gt.shape[0],), device=device)
            
            # 这个不是真loss，这个是 forward+loss 这里面的forward是黄色部分的，要训练，需要存储中间变量和特征
            loss = diffusion.p_losses(x_start = z_gt, cond = cond, t = t) 


            # opt_cond.zero_grad()
            opt_diff.zero_grad()
            accelerator.backward(loss)
            # opt_cond.step()
            opt_diff.step()

            accelerator.wait_for_everyone()

            global_step += 1
            step_loss.append(loss.item())
            epoch_loss.append(loss.item())
            pbar.update(1)
            pbar.set_description(f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, Loss: {loss.item():.6f}")

            # Log loss values:
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                # Gather values from all processes
                avg_loss = accelerator.gather(torch.tensor(step_loss, device=device).unsqueeze(0)).mean().item()
                step_loss.clear()
                if accelerator.is_local_main_process:
                    writer.add_scalar("loss/loss_simple_step", avg_loss, global_step)

            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_local_main_process:
                    checkpoint = diffusion.controlnet.state_dict()
                    ckpt_path = f"{ckpt_dir}/{global_step:07d}_controlnet.pt"
                    torch.save(checkpoint, ckpt_path)
                    checkpoint = cond_pretrain.state_dict()
                    ckpt_path = f"{ckpt_dir}/{global_step:07d}_cond.pt"
                    torch.save(checkpoint, ckpt_path)

            if global_step % cfg.train.image_every == 0 or global_step == 1:
                N = 1
                log_gt, log_lq = gt[:N], lq[:N]
                # log_gt = gt[:N]
                diffusion.controlnet.eval()
                cond_pretrain.eval()
                # 在下面测试的时候，全都with no grad
                # z_1 = vae.encoder(log_gt)
                # cond, latent_variable = setup_noise_inputs(device=device, img = z_1, batchsize = 2) 
                with torch.no_grad():
                    synth_img = sample_from_ddpm(
                    ddpm = diffusion,
                    decoder = vae.reconstruct_ldm_outputs,
                    img=latent_variable,
                    device=accelerator.device,
                    cond=cond,
                    num_timesteps=1000
                )
                    
                    if accelerator.is_local_main_process:
                        x_tb = list()
                        y_tb = list()
                        z_tb = list()

                        for tag, image in [
                            ("image/samples", synth_img),
                            ("image/gt", log_gt),
                            ("image/lq", log_lq),
                        ]:
                            image = image.detach().cpu()
                            # Check if the image is a 3D MRI image
                            slices = get_middle_slice(image[0].squeeze())  # remove channel dimension and get slices
                            # for i, slice_img in enumerate(slices):
                            x_tb.append(slices[0].unsqueeze(0).unsqueeze(0))
                            y_tb.append(slices[1].unsqueeze(0).unsqueeze(0))
                            z_tb.append(slices[2].unsqueeze(0).unsqueeze(0))
                            
                        # 将每个内部列表拼接
                        x_tb_concat = [torch.cat([x], dim=0) for x in x_tb]
                        x_ = torch.cat(x_tb_concat, dim=0)
                        y_tb_concat = [torch.cat([y], dim=0) for y in y_tb]
                        y_ = torch.cat(y_tb_concat, dim=0)
                        z_tb_concat = [torch.cat([z], dim=0) for z in z_tb]
                        z_ = torch.cat(z_tb_concat, dim=0)

                        writer.add_image(f"slice/x", make_grid(x_, nrow=3), global_step)
                        writer.add_image(f"slice/y", make_grid(y_, nrow=3), global_step)
                        writer.add_image(f"slice/z", make_grid(z_, nrow=3), global_step)

                    diffusion.controlnet.train()
                    # cond_pretrain.train()
            


            if global_step % cfg.train.val_every == 0 and global_step > 0:
                diffusion.controlnet.eval()
                cond_pretrain.eval()
                val_loss = []
                clean_psnr = []
                val_lpips = []
                val_psnr = []
                val_pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch", 
                                total=len(val_loader), leave=False, desc="Validation")
                
                for val_gt, val_lq, lq_grad in val_loader:
                    val_gt = rearrange(val_gt, "b h w z c -> b c h w z").contiguous().float().to(device)
                    val_lq = rearrange(val_lq, "b h w z c -> b c h w z").contiguous().float().to(device)
                    lq_grad = rearrange(lq_grad, "b h w z c -> b c h w z").contiguous().float().to(device)
                    

                    with torch.no_grad():
                        z_gt = vae.encode(val_gt)
                        # print(z_gt.shape)
                        z_lq = vae.encode(val_lq)
                        z_lq_grad = grad_vae.encode(lq_grad)
                        z_hit = torch.cat([z_lq, z_lq_grad], dim=1)

                        c_txt = cond_pretrain(val_lq)
                        # c_txt = torch.tensor([[0.5, 0.5, 0.2, 0.5]],device=device)
                        cond, latent_variable = setup_noise_inputs(device=device, img = z_hit, batchsize=1, c_txt = c_txt) 
                    t = torch.randint(0, diffusion.num_timesteps, (z_gt.shape[0],), device=device)
                    
                    with torch.no_grad():
                        loss = diffusion.p_losses(x_start = z_gt, cond = cond, t = t) 

                    # accelerator.wait_for_everyone()
                    # print(f"=============={cond["c_img"].shape}=================")
                    with torch.no_grad():
                        synth_img = sample_from_ddpm(
                                ddpm = diffusion,
                                decoder = vae.reconstruct_ldm_outputs,
                                img=latent_variable,
                                device=accelerator.device,
                                cond=cond,
                                num_timesteps=1000
                            )
                    # 注意，以下转换中假设val_gt和val_lq的shape是[B, H, W, 1]，灰度图1个通道
                    # 由于灰度图像只有一个通道，这里将最后一个维度去除了
                    
                    with torch.no_grad():
                        # forward
                        # val_pred = pred 
                        val_pred = synth_img # SwinIR预计一个通道维度，对于灰度图，需要unsqueeze(1)添加通道维度
                        # compute metrics (loss, lpips, psnr)
                        val_loss.append(loss.item())  # 对于灰度图也需要添加unsqueeze(1)
                        # 请确保lpips_model可以接受单通道输入，这里添加unsqueeze(1)用于确保形状兼容
                        # val_lpips.append(lpips_model(val_pred, val_gt.unsqueeze(1), normalize=True).mean().item())
                        # 对于PSNR，需要确保输入值的形状兼容

                        val_psnr.append(calculate_psnr_pt(val_pred, val_gt , crop_border=0).mean().item())
                    val_pbar.update(1)

                    if accelerator.is_local_main_process:
                        for tag, image in [
                            ("image/samples", synth_img),
                            ("image/gt", val_gt),
                            ("image/lq", val_lq),
                        ]:
                            image = image.detach().cpu()
                            # Check if the image is a 3D MRI image
                            slices = get_middle_slice(image[0].squeeze())  # remove channel dimension and get slices
                            # for i, slice_img in enumerate(slices):
                            x_tb.append(slices[0].unsqueeze(0).unsqueeze(0))
                            y_tb.append(slices[1].unsqueeze(0).unsqueeze(0))
                            z_tb.append(slices[2].unsqueeze(0).unsqueeze(0))
                            
                        # 将每个内部列表拼接
                        x_tb_concat = [torch.cat([x], dim=0) for x in x_tb]
                        x_ = torch.cat(x_tb_concat, dim=0)
                        y_tb_concat = [torch.cat([y], dim=0) for y in y_tb]
                        y_ = torch.cat(y_tb_concat, dim=0)
                        z_tb_concat = [torch.cat([z], dim=0) for z in z_tb]
                        z_ = torch.cat(z_tb_concat, dim=0)

                        writer.add_image(f"val/x", make_grid(x_, nrow=3), global_step)
                        writer.add_image(f"val/y", make_grid(y_, nrow=3), global_step)
                        writer.add_image(f"val/z", make_grid(z_, nrow=3), global_step)
                    
                    # cond_pretrain.train()
                    diffusion.controlnet.train()
                            
                val_pbar.close()
                avg_val_loss = accelerator.gather(torch.tensor(val_loss, device=device).unsqueeze(0)).mean().item()
                # avg_val_lpips = accelerator.gather(torch.tensor(val_lpips, device=device).unsqueeze(0)).mean().item()
                avg_val_psnr = accelerator.gather(torch.tensor(val_psnr, device=device).unsqueeze(0)).mean().item()
            
                if accelerator.is_local_main_process:
                    for tag, val in [
                        ("val/loss", avg_val_loss),
                        ("val/psnr", avg_val_psnr)
                    ]:
                        writer.add_scalar(tag, val, global_step)
                    writer.add_text('cond', str(c_txt), global_step)
                    
                diffusion.controlnet.train()
                # cond_pretrain.train()

            accelerator.wait_for_everyone()

            if global_step == max_steps:
                break
        
        pbar.close()
        epoch += 1
        # scheduler_cond.step()
        avg_epoch_loss = accelerator.gather(torch.tensor(epoch_loss, device=device).unsqueeze(0)).mean().item()
        epoch_loss.clear()
        if accelerator.is_local_main_process:
            writer.add_scalar("loss/loss_simple_epoch", avg_epoch_loss, global_step)

    if accelerator.is_local_main_process:
        print("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
