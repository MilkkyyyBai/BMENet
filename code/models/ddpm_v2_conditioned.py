from functools import partial
from inspect import isfunction
from typing import Dict, Tuple, Set, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

from models.unet_v2_conditioned import UNetModel
from models.controlent_new import ControlNet
from models.util import (
    timestep_embedding,
    exists
)
from project.utils.cond_fn import *


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(
        shape[0], *((1,) * (len(shape) - 1))
    )
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def make_beta_schedule(
    schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "linear":
        betas = (
            torch.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=torch.float64
            )
            ** 2
        )

    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(
            linear_start, linear_end, n_timestep, dtype=torch.float64
        )
    elif schedule == "sqrt":
        betas = (
            torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
            ** 0.5
        )
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()

def disabled_train(self: nn.Module) -> nn.Module:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class DDPM(nn.Module):
    def __init__(
        self,
        unet_config,
        cldm_config,
        timesteps: int = 1000,
        beta_schedule="linear",
        loss_type="l2",
        log_every_t=100,
        clip_denoised=False,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        original_elbo_weight=0.0,
        v_posterior=0.0,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        l_simple_weight=1.0,
        parameterization="eps",  # all assuming fixed variance schedules
        learn_logvar=False,
        logvar_init=0.0,
        conditioning_key=None,
        vae=None
    ):
        super().__init__()
        assert parameterization in [
            "eps",
            "x0",
        ], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization

        if conditioning_key == "unconditioned":
            conditioning_key = None
        self.conditioning_key = conditioning_key
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        
        self.context={}
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        self.loss_type = loss_type

        self.register_schedule(
            beta_schedule=beta_schedule,
            timesteps=timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)
        self.cldm_config = cldm_config
        self.controlnet = ControlNet(**self.cldm_config.get("params", dict())).to('cuda')

    def setVAE(self, vae):
        self.vae = vae

    def load_pretrained_sd(self, sd: Dict[str, torch.Tensor]) -> Set[str]:
        module_map = {
            "model": "model",
        }
        modules = [("model", self.model)]
        used = set()
        for name, module in modules:
            init_sd = {}
            scratch_sd = module.state_dict()
            for key in scratch_sd:
                target_key = ".".join([module_map[name], key])
                init_sd[key] = sd[target_key].clone()
                used.add(target_key)
            module.load_state_dict(init_sd, strict=True)
        unused = used
        # NOTE: this is slightly different from previous version, which haven't switched
        # the UNet to eval mode and disabled the requires_grad flag.
        for module in [self.model]:
            module.eval()
            module.train = disabled_train
            for p in module.parameters():
                p.requires_grad = False
        return unused
    
    @torch.no_grad()    
    def load_controlnet_from_ckpt(self, sd: Dict[str, torch.Tensor]) -> None:
        self.controlnet.load_state_dict({k.replace('module.',''):v for k,v in sd.items()}, strict=True)

    def register_schedule(
        self,
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
    ):
        betas = make_beta_schedule(
            beta_schedule,
            timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (
            1.0 - alphas_cumprod_prev
        ) / (1.0 - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

        if self.parameterization == "eps":
            lvlb_weights = self.betas**2 / (
                2
                * self.posterior_variance
                * to_torch(alphas)
                * (1 - self.alphas_cumprod)
            )
        elif self.parameterization == "x0":
            lvlb_weights = (
                0.5
                * np.sqrt(torch.Tensor(alphas_cumprod))
                / (2.0 * 1 - torch.Tensor(alphas_cumprod))
            )
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer("lvlb_weights", lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
        q(x_{t-1} | x_t, x_0)
        """
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def apply_cond_fn(
        self,
        pred_x0: torch.Tensor,
        t: torch.Tensor,
        index: torch.Tensor,
        cond_fn: Guidance
    ) -> torch.Tensor:
        t_now = int(t[0].item()) + 1
        if not (cond_fn.t_stop < t_now and t_now < cond_fn.t_start):
            # stop guidance
            self.context["g_apply"] = False
            return pred_x0
        grad_rescale = 1 / extract(self.posterior_mean_coef1, index, pred_x0.shape)
        # apply guidance for multiple times
        loss_vals = []
        for _ in range(cond_fn.repeat):
            # set target and pred for gradient computation
            target, pred = None, None
            if cond_fn.space == "latent":
                target = self.vae.encode(cond_fn.target)
                pred = pred_x0
            elif cond_fn.space == "rgb":
                # We need to backward gradient to x0 in latent space, so it's required
                # to trace the computation graph while decoding the latent.
                with torch.enable_grad():
                    target = cond_fn.target
                    pred_x0_rg = pred_x0.detach().clone().requires_grad_(True)
                    pred = self.vae.reconstruct_ldm_outputs(pred_x0_rg)
                    assert pred.requires_grad
            else:
                raise NotImplementedError(cond_fn.space)
            # compute gradient
            delta_pred, loss_val = cond_fn(target, pred, t_now)
            loss_vals.append(loss_val)
            # update pred_x0 w.r.t gradient
            if cond_fn.space == "latent":
                delta_pred_x0 = delta_pred
                pred_x0 = pred_x0 + delta_pred_x0 * grad_rescale
            elif cond_fn.space == "rgb":
                pred.backward(delta_pred)
                delta_pred_x0 = pred_x0_rg.grad
                pred_x0 = pred_x0 + delta_pred_x0 * grad_rescale
            else:
                raise NotImplementedError(cond_fn.space)
        self.context["g_apply"] = True
        self.context["g_loss"] = float(np.mean(loss_vals))
        return pred_x0


    def p_mean_variance(self, x, c, t, index, clip_denoised: bool, return_x0=False, cond_fn: Guidance=None):
        """
        Apply the model to get p(x_{t-1} | x_t)
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].

        """
        t_in = t
        model_out = self.apply_model(x, t_in, c) # 进这里面
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out

        if cond_fn:
            x_recon = self.apply_cond_fn(x_recon, t, index, cond_fn)
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    def p_sample(
        self,
        x,
        c,
        t,
        index,
        clip_denoised=True,
        repeat_noise=False,
        return_x0=False,
        temperature=1.0,
        noise_dropout=0.0,
        cond_fn: Guidance=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param clip_denoised: if True, clip the x_start prediction to [-1, 1].
        """

        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(
            x=x,
            c=c, # 进这里面
            t=t,
            index=index,
            clip_denoised=clip_denoised,
            return_x0=return_x0,
            cond_fn=cond_fn
        )
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.0:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        if return_x0:
            return self.context, (
                model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise,
                x0,
            )
        else:
            return self.context, model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_sample_loop(self, cond, shape, return_intermediates=False):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]

        for i in tqdm(
            reversed(range(0, self.num_timesteps)),
            desc="sampling loop time step",
            total=self.num_timesteps,
        ):
            img = self.p_sample(
                img,
                cond,
                torch.full((b,), i, device=device, dtype=torch.long),
                clip_denoised=self.clip_denoised,
            )
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop(
            (batch_size, channels, image_size, image_size),
            return_intermediates=return_intermediates,
        )

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == "l1":
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == "ncc":
            ncc_loss = NCCLoss()
            loss = ncc_loss.loss(target, pred)
        elif self.loss_type == "l2":
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction="none")
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, cond, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        cond_img = cond['c_img']
        cond_txt = cond['c_txt']  # this

        cond_tmp = cond_txt.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # shape: [1, 4, 1, 1, 1]
        cond_crossatten = (
            cond_tmp.squeeze(-1).squeeze(-1).squeeze(-1).unsqueeze(1)
        )  # shape: [1, 1, 4]  条件一步一步到这 变成cond_crossatten, 放在cond_concat里面
        cond_concat = torch.tile(cond_tmp, (20, 28, 20))
        # batchsize = 2
        # for i in range(batchsize - 1):
        #     cond_concat = torch.cat((cond_concat, cond_concat), dim=0)

        conditioning = {
            "c_concat": [cond_concat],
            "c_crossattn": cond_crossatten,
            "c_img": cond_img
        }


        model_output = self.apply_model(x_noisy, t, conditioning)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(
                f"Paramterization {self.parameterization} not yet supported"
            )
        # loss_simple = self.get_loss(model_output, target, mean=False).mean(
        #     dim=[1, 2, 3, 4]
        # )
        loss_simple = self.get_loss(model_output, target, mean=True)
        # print(loss_simple.shape)
        # loss_dict.update({f"loss_simple": loss_simple.mean()})

        # logvar_t = self.logvar[t].to(x_start.device)
        # loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        # if self.learn_logvar:
        #     loss_dict.update({f"loss_gamma": loss.mean()})
        #     loss_dict.update({"logvar": self.logvar.data.mean()})

        # loss = self.l_simple_weight * loss.mean()

        # loss_vlb = self.get_loss(model_output, target, mean=False).mean(
        #     dim=(1, 2, 3, 4)
        # )
        # loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        # loss_dict.update({f"loss_vlb": loss_vlb})
        # loss += self.original_elbo_weight * loss_vlb
        # loss_dict.update({f"loss": loss})

        return loss_simple

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=x.device).long()
        return self.p_losses(x, c, t, *args, **kwargs)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            print("Diffusion model optimizing logvar")
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

    def apply_model(self, x_noisy, t, cond, return_ids=False):
   
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = (
                "c_concat" if self.model.conditioning_key == "concat" else "c_crossattn"
            )
            cond = {key: cond}

        # x_recon = self.model(x_noisy, t, **cond)
        

        c_txt = cond["c_crossattn"]
        c_img = cond["c_img"]
        c_concat = cond['c_concat']



        control = self.controlnet(
            x=x_noisy, hint=c_img,
            timesteps=t,  context=None
        )
        control_scales = [1.0] * 13
        control = [c * scale for c, scale in zip(control, control_scales)]


        x_recon = self.model(
            x=x_noisy, timesteps=t,
            context=c_txt, control=control, only_mid_control=False, c_concat=c_concat
        )

        # eps = self.unet(
        #     x=x_noisy, timesteps=t,
        #     context=c_txt, control=control, only_mid_control=False
        # )

        # if isinstance(x_recon, tuple) and not return_ids:
        #     return x_recon[0]
        # else:
        return x_recon


class DiffusionWrapper(nn.Module):
    def __init__(self, unet_config, conditioning_key):
        super().__init__()
        self.diffusion_model = ControlledUnetModel(**unet_config.get("params", dict()))
        # self.diffusion_model = UNetModel()
        self.conditioning_key = conditioning_key

    # def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):  
    def forward(self, x, timesteps, context, control, only_mid_control, c_concat):
        # 你把 ControlledUnetModel 封装进了 Diffusion Wrapper， 但现在你得forwad函数，入参也没改，496行的out也没有变，它对不上呀
        xc = torch.cat([x] + c_concat, dim=1)  # TODO:Can we update this latent space?
        # cc = torch.cat(c_crossattn, 1)
        # out = self.diffusion_model(xc, t, context=cc)
        out = self.diffusion_model(x=xc, timesteps=timesteps,
            context=context, control=control, only_mid_control=only_mid_control)



        return out
    


class ControlledUnetModel(UNetModel):

    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):
        hs = []
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.time_embed(t_emb)
        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb, context)
            hs.append(h)
        h = self.middle_block(h, emb, context)

        if control is not None:
            h += control.pop()

        for i, module in enumerate(self.output_blocks):
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)
            h = module(h, emb, context)

        h = h.type(x.dtype)
        return self.out(h)

import math

class NCCLoss:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        # print(y_pred.shape)
        # print(y_true.shape)
        I = y_true
        J = y_pred

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(I.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 3, *win]).to(y_true.device)

        pad_no = math.floor(win[0]/2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1,1)
            padding = (pad_no, pad_no)
        else:
            stride = (1,1,1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        I_sum = conv_fn(I, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(J, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)
        # return -cc


import torch
import torch.nn as nn
from torch.autograd import Variable

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

def main():
    # x = Variable(torch.FloatTensor([[[1,2],[2,3]],[[1,2],[2,3]]]).view(1,2,2,2), requires_grad=True)
    # x = Variable(torch.FloatTensor([[[3,1],[4,3]],[[3,1],[4,3]]]).view(1,2,2,2), requires_grad=True)
    # x = Variable(torch.FloatTensor([[[1,1,1], [2,2,2],[3,3,3]],[[1,1,1], [2,2,2],[3,3,3]]]).view(1, 2, 3, 3), requires_grad=True)
    x = Variable(torch.FloatTensor([[[1, 2, 3], [2, 3, 4], [3, 4, 5]], [[1, 2, 3], [2, 3, 4], [3, 4, 5]]]).view(1, 2, 3, 3),requires_grad=True)
    addition = TVLoss()
    z = addition(x)
    # print x
    # print z.data
    # z.backward()
    # print x.grad
    

