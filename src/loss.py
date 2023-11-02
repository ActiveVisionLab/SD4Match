import torch
import torch.nn.functional as F
from utils.geometry import *
from utils.misc import *
from utils.matching import L2normalization
import numpy as np
from typing import List

class GaussianCrossEntropyLoss:     # Change the name to FeatMapLoss

    def __init__(self):
        pass

    def __call__(self, src_featmaps, trg_featmaps, src_kps, trg_kps, src_imgsize, trg_imgsize, npts, softmax_temp, enable_l2_norm, **kwargs):
 
        bz, c, h1, w1 = trg_featmaps.shape

        loss = 0
        for b in range(bz):

            featmap0 = src_featmaps[b:b+1].clone()
            featmap1 = trg_featmaps[b:b+1].clone()
            xy0 = src_kps[b:b+1].clone()
            xy1 = trg_kps[b:b+1].clone()
            npt = npts[b].clone()

            # normalize xy0 to [-1, 1] and scale xy1 t0 (h1, w1)
            xy0 = normalise_coordinates(xy0, src_imgsize)
            xy1 = scaling_coordinates(xy1, trg_imgsize, (h1, w1))
            # filter out invalid pointloss_fn
            xy0 = xy0.view(-1, 2)[:npt].unsqueeze(0)  # 1 x N x 2
            xy1 = xy1.view(-1, 2)[:npt].unsqueeze(0)  # 1 x N x 2

            # extract features of xy0
            xy0_feat = F.grid_sample(featmap0, xy0.unsqueeze(2), mode='bilinear', align_corners=True)   # 1 x c x N x 1
            xy0_feat = xy0_feat.squeeze(-1).permute(0, 2, 1)    # 1 x N x c

            # calculate correlation score
            if enable_l2_norm:
                xy0_feat = L2normalization(xy0_feat, dim=-1)
                featmap1 = L2normalization(featmap1, dim=1).reshape(1, c, -1)   # 1 x c x (h1w1)
            else:
                featmap1 = featmap1.reshape(1, c, -1)

            xy0_corr = torch.bmm(xy0_feat, featmap1)    # 1 x N x (h1w1)
            xy0_corr = softmax_with_temperature(xy0_corr, softmax_temp, dim=-1).reshape(1, -1, h1, w1)

            loss = loss + self.efficient_gaussian_smoothed_cross_entropy(xy0_corr, xy1, 7)
        
        loss = loss / bz

        return loss


    def efficient_gaussian_smoothed_cross_entropy(self, xy0_corr, xy1, gau_ker_size):
        '''
        IN:
            xy0_corr[torch.Tensor](B x N x h2 x w2)
            kps[torch.Tensor](B x N x 2): correspondences, in (x1, y1) in featmap1 scale (h1, w1)
            gau_ker_size[int]: size of gaussian kernel
        '''
        B, N, h1, w1 = xy0_corr.shape
        _device = xy0_corr.device

        # use bilinear sampling to extract kernel area
        # find out gaussian kernel convered position
        ind = gau_ker_size // 2
        kx = torch.linspace(-ind, ind, gau_ker_size).to(_device)
        ky = torch.linspace(-ind, ind, gau_ker_size).to(_device)
        ky, kx = torch.meshgrid(ky, kx)
        kernel = torch.stack([kx, ky], dim=-1)
        kernel = kernel.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1, -1)     # B x N x gau_ker_size x gau_ker_size x 2

        xy1 = xy1.unsqueeze(-2).unsqueeze(-2).expand(-1, -1, gau_ker_size, gau_ker_size, -1)    # B x N x gau_ker_size x gau_ker_size x 2
        xy1 = xy1 + kernel
        mask = (xy1[..., 0] > 0) & (xy1[..., 0] < w1-1) & (xy1[..., 1] > 0) & (xy1[..., 1] < h1-1)

        # extract value
        xy1 = normalise_coordinates(xy1, (h1, w1)).view(-1, gau_ker_size, gau_ker_size, 2)
        xy0_corr = xy0_corr.view(-1, h1, w1).unsqueeze(1)
        scores = F.grid_sample(xy0_corr, xy1, mode='bilinear', padding_mode='border', align_corners=True)
        scores = scores.view(B, N, gau_ker_size, gau_ker_size)

        # multiply kernel with scores
        kernel = self.gaussian_kernel_generator(gau_ker_size, _device)
        kernel = kernel.unsqueeze(0).unsqueeze(0).expand(B, N, -1, -1)
        loss = -(kernel * torch.log(scores) * mask).sum() / (B * N)

        return loss


    def gaussian_kernel_generator(self, size, _device):
        '''
        Generate a size x size 2D gaussian kernel
        '''
        if size > 1:
            ind = size // 2
            kx = torch.linspace(-ind, ind, size).to(_device)
            ky = torch.linspace(-ind, ind, size).to(_device)

            ky, kx = torch.meshgrid(ky, kx)
            kernel = torch.sqrt(kx**2 + ky**2)

            mu, sigma = 0, ind/2
            kernel = 1 / (sigma**2*2*np.pi) * torch.exp(-1/2*((kernel-mu)/sigma)**2)
        else:
            kernel = torch.ones((1, 1), device=_device)

        return kernel