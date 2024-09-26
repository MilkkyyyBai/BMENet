from typing import overload, Tuple
import torch
from torch.nn import functional as F
import math
import numpy as np
class Guidance:

    def __init__(self, scale: float, t_start: int, t_stop: int, space: str, repeat: int) -> "Guidance":
        """
        Initialize restoration guidance.

        Args:
            scale (float): Gradient scale (denoted as `s` in our paper). The larger the gradient scale, 
                the closer the final result will be to the output of the first stage model.
            t_start (int), t_stop (int): The timestep to start or stop guidance. Note that the sampling 
                process starts from t=1000 to t=0, the `t_start` should be larger than `t_stop`.
            space (str): The data space for computing loss function (rgb or latent).

        Our restoration guidance is based on [GDP](https://github.com/Fayeben/GenerativeDiffusionPrior).
        Thanks for their work!
        """
        self.scale = scale
        self.t_start = t_start
        self.t_stop = t_stop
        self.target = None
        self.space = space
        self.repeat = repeat
    
    def load_target(self, target: torch.Tensor) -> None:
        self.target = target

    def __call__(self, target_x0: torch.Tensor, pred_x0: torch.Tensor, t: int) -> Tuple[torch.Tensor, float]:
        # avoid propagating gradient out of this scope
        pred_x0 = pred_x0.detach().clone()
        target_x0 = target_x0.detach().clone()
        return self._forward(target_x0, pred_x0, t)
    
    @overload
    def _forward(self, target_x0: torch.Tensor, pred_x0: torch.Tensor, t: int) -> Tuple[torch.Tensor, float]:
        ...


class MSEGuidance(Guidance):

    def _forward(self, target_x0: torch.Tensor, pred_x0: torch.Tensor, t: int) -> Tuple[torch.Tensor, float]:
        # inputs: [-1, 1], nchw, rgb
        with torch.enable_grad():
            pred_x0.requires_grad_(True)
            loss = (pred_x0 - target_x0).pow(2).mean((1, 2, 3, 4)).sum()
        scale = self.scale
        g = -torch.autograd.grad(loss, pred_x0)[0] * scale
        return g, loss.item()


class WeightedMSEGuidance(Guidance):

    def _get_weight(self, target: torch.Tensor) -> torch.Tensor:
        # convert RGB to G
        rgb_to_gray_kernel = torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
        target = torch.sum(target * rgb_to_gray_kernel.to(target.device), dim=1, keepdim=True)
        # initialize sobel kernel in x and y axis
        G_x = [
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ]
        G_y = [
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ]
        G_x = torch.tensor(G_x, dtype=target.dtype, device=target.device)[None]
        G_y = torch.tensor(G_y, dtype=target.dtype, device=target.device)[None]
        G = torch.stack((G_x, G_y))

        target = F.pad(target, (1, 1, 1, 1), mode='replicate') # padding = 1
        grad = F.conv2d(target, G, stride=1)
        mag = grad.pow(2).sum(dim=1, keepdim=True).sqrt()

        n, c, h, w = mag.size()
        block_size = 2
        blocks = mag.view(n, c, h // block_size, block_size, w // block_size, block_size).permute(0, 1, 2, 4, 3, 5).contiguous()
        block_mean = blocks.sum(dim=(-2, -1), keepdim=True).tanh().repeat(1, 1, 1, 1, block_size, block_size).permute(0, 1, 2, 4, 3, 5).contiguous()
        block_mean = block_mean.view(n, c, h, w)
        weight_map = 1 - block_mean

        return weight_map

    def _forward(self, target_x0: torch.Tensor, pred_x0: torch.Tensor, t: int) -> Tuple[torch.Tensor, float]:
        # inputs: [-1, 1], nchw, rgb
        with torch.no_grad():
            w = self._get_weight((target_x0 + 1) / 2)
        with torch.enable_grad():
            pred_x0.requires_grad_(True)
            loss = ((pred_x0 - target_x0).pow(2) * w).mean((1, 2, 3)).sum()
        scale = self.scale
        g = -torch.autograd.grad(loss, pred_x0)[0] * scale
        return g, loss.item()

class NCCGuidace(Guidance):
    
    def __init__(self, scale: float, t_start: int, t_stop: int, space: str, repeat: int) -> "Guidance":
        super(NCCGuidace, self).__init__(scale, t_start, t_stop, space, repeat)
        self.ncc_loss = NCCLoss()

    def _forward(self, target_x0: torch.Tensor, pred_x0: torch.Tensor, t: int) -> Tuple[torch.Tensor, float]:
        # inputs: [-1, 1], nchw, rgb
        # print(target_x0.device)
        with torch.enable_grad():
            pred_x0.requires_grad_(True)
            # loss = ((pred_x0 - target_x0).pow(2) * w).mean((1, 2, 3)).sum()
            cc = self.ncc_loss.loss(target_x0, pred_x0)
            loss = cc.mean((1, 2, 3, 4)).sum()

        scale = self.scale
        g = -torch.autograd.grad(loss, pred_x0)[0] * scale
        return g, loss.item()

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
        sum_filt = torch.ones([1, 1, *win]).to(y_true.device)

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

        return cc
        # return -cc
