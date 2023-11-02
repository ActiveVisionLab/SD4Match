import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

from .geometry import *
from .misc import *


def L2normalization(feat, dim):

    return feat / (torch.norm(feat, dim=dim, keepdim=True) + 1e-6)


def extract_feature(featmap, kps):
    '''
    IN:
        featmap [torch.Tensor] (B x C x H x W)
        kps [torch.Tensor] (B x ... x 2): It should be in featmap's scale
    OUT:
        feat [torch.Tensor] (B x ... x C)
    '''
    H, W = featmap.shape[2:]
    output_shape = list(kps.shape)
    output_shape[-1] = featmap.shape[1]
    output_shape = tuple(output_shape)

    kps_ = normalise_coordinates(kps, (H, W))
    kps_ = kps_.reshape(output_shape[0], -1, 2)
    kps_ = kps_[:, :, None, :]
    feat = nn.functional.grid_sample(featmap, kps_, align_corners=True)

    return feat.permute(0, 2, 3, 1).reshape(output_shape)


def select_top_feat_coord(feat, featmap, metric='cosine', l2_norm=True):
    '''
    IN:
        feat [torch.Tensor] (B x ... x C)
        featmap [torch.Tensor] (B x C x H x W)
    OUT:
        coord [torch.Tensor] (B x ... x 2)
    '''

    # normalise feat and featmap
    if l2_norm:
        feat = L2normalization(feat, dim=-1)
        featmap = L2normalization(featmap, dim=1)

    b, c, h, w = featmap.shape
    featmap = featmap.view(b, c, -1).permute(0, 2, 1)

    base_shape = tuple(feat.shape[:-1])
    output_shape = base_shape + (2,)
    feat = feat.view(b, -1, c)
    
    if metric == 'L2':
        feat = feat[:, :, None, :]
        featmap = featmap[:, None, :, :]
        dist = torch.norm(feat-featmap, dim=-1)
        coord = dist.argsort(dim=-1)
        coord = coord[:, :, 0]        
    elif metric == 'cosine':
        featmap = featmap.permute(0, 2, 1)
        similarity = torch.bmm(feat, featmap)
        coord = similarity.argsort(dim=-1, descending=True)
        coord = coord[:, :, 0]

    y = coord // w
    x = coord % w

    coord = torch.stack((x, y), dim=-1)
    coord = coord.float()
    coord = coord.view(output_shape)

    return coord


def nn_get_matches(src_featmaps, trg_featmaps, query, l2_norm=True):
    '''
    Find the top k matches by nearest neighbour of query. 
    IN:
        src_featmaps [torch.Tensor] (B x C x H x W): feature map of the source image from where the features of query are extracted
        trg_featmaps [torch.Tensor] (B x C x H x W): feature map of the target image. The matches are selected from this feature map
        query [torch.Tensor] (B x ... x 2): query point. Note that query point MUST BE IN the scale src_featmaps NOT IN the scale of image
    OUT:
        coord [torch.Tensor] (B x ... x 2)
    '''
    feat_query = extract_feature(src_featmaps, query)
    
    coord = select_top_feat_coord(feat_query, trg_featmaps, l2_norm=l2_norm)

    return coord


def bilinear_get_matches(src_featmaps, trg_featmaps, query, l2_norm=True):
    '''
    IN:
    Find the top k matches by four corner bilinear interpolation
        src_featmaps [torch.Tensor] (B x C x H x W)
        trg_featmaps [torch.Tensor] (B x C x H x W)
        query [torch.Tensor] (B x ... x 2): query point. Note that query point MUST BE IN the scale src_featmaps NOT IN the scale of image
        top_k [Int]
    OUT:
        matches [torch.Tensor] (B x ... x 2)
    '''

    h1, w1 = src_featmaps.shape[2:]
    h2, w2 = trg_featmaps.shape[2:]

    # push coordinate into image boundary
    query = regularise_coordinates(query, h1, w1, eps=1e-5)

    i_m_m = torch.cat((torch.floor(query[..., 0:1]), torch.floor(query[..., 1:2])), dim=-1)    # index_minus_minus (x, y)
    i_m_p = torch.cat((torch.floor(query[..., 0:1]), torch.floor(query[..., 1:2])+1), dim=-1)     # index_minus_plus
    i_p_m = torch.cat((torch.floor(query[..., 0:1])+1, torch.floor(query[..., 1:2])), dim=-1)     # index_plus_minus
    i_p_p = torch.cat((torch.floor(query[..., 0:1])+1, torch.floor(query[..., 1:2])+1), dim=-1)      # index_plus_plus

    multrows = lambda x: x[..., 0:1] * x[..., 1:2]
    f_m_m = multrows(torch.abs(query - i_p_p))
    f_m_p = multrows(torch.abs(query - i_p_m))
    f_p_m = multrows(torch.abs(query - i_m_p))
    f_p_p = multrows(torch.abs(query - i_m_m))

    Q_m_m = nn_get_matches(src_featmaps, trg_featmaps, i_m_m, l2_norm)
    Q_m_p = nn_get_matches(src_featmaps, trg_featmaps, i_m_p, l2_norm)
    Q_p_m = nn_get_matches(src_featmaps, trg_featmaps, i_p_m, l2_norm)
    Q_p_p = nn_get_matches(src_featmaps, trg_featmaps, i_p_p, l2_norm)

    matches = (Q_m_m*f_m_m+Q_p_p*f_p_p+Q_m_p*f_m_p+Q_p_m*f_p_m)/(f_p_p+f_m_m+f_m_p+f_p_m)   # matches in feature-resolution

    return matches


def softargmax_get_matches(src_featmaps, trg_featmaps, query, softmax_temp, l2_norm=True):
    '''
    IN:
    Find the top k matches by four corner bilinear interpolation
        src_featmaps [torch.Tensor] (B x C x H x W)
        trg_featmaps [torch.Tensor] (B x C x H x W)
        query [torch.Tensor] (B x Nq x 2): query point. Note that query point MUST BE IN the scale src_featmaps NOT IN the scale of image
        softmax_temp [float] : temperature for softmax. If 'src_temp' is in batch (which indicate auto temp), this would be ignore.
    OUT:
        matches [torch.Tensor] (B x Nq x 2)
    '''
    B, C, h1, w1 = src_featmaps.shape
    h2, w2 = trg_featmaps.shape[2:]
    Nq = query.shape[1]
    _device = src_featmaps.device

    # extract query's feature
    feat_query = extract_feature(src_featmaps, query)   # B x Nq x C

    # calculate score map
    if l2_norm:
        feat_query = L2normalization(feat_query, dim=-1)
        trg_featmaps = L2normalization(trg_featmaps, dim=1)

    scoremaps = feat_query @ trg_featmaps.view(B, C, -1)
    scoremaps = softmax_with_temperature(scoremaps, softmax_temp, -1)

    # create grid
    grid = create_grid(h2, w2, device=_device)

    # average over grid
    scoremaps = scoremaps.unsqueeze(-1).expand(-1, -1, -1, 2)
    grid = grid.view(-1, 2).unsqueeze(0).unsqueeze(0).expand(B, Nq, -1, -1)
    matches = (scoremaps * grid).sum(dim=2)

    return matches


def kernel_softargmax_get_matches(src_featmaps, trg_featmaps, query, softmax_temp, sigma=7, l2_norm=True):
    '''
    IN:
    Find the top k matches by four corner bilinear interpolation
        src_featmaps [torch.Tensor] (B x C x H x W)
        trg_featmaps [torch.Tensor] (B x C x H x W)
        query [torch.Tensor] (B x Nq x 2): query point. Note that query point MUST BE IN the scale src_featmaps NOT IN the scale of image
        softmax_temp [float]: temperature for softmax. If 'src_temp' is in batch (which indicate auto temp), this would be ignore.
        sigma [int]: sigma of the gaussian suppression
    OUT:
        matches [torch.Tensor] (B x Nq x 2)
    '''
    B, C, h1, w1 = src_featmaps.shape
    h2, w2 = trg_featmaps.shape[2:]
    Nq = query.shape[1]
    _device = src_featmaps.device

    # extract query's feature
    feat_query = extract_feature(src_featmaps, query)   # B x Nq x C

    # calculate score map
    if l2_norm:
        feat_query = L2normalization(feat_query, dim=-1)
        trg_featmaps = L2normalization(trg_featmaps, dim=1)

    scoremaps = feat_query @ trg_featmaps.view(B, C, -1)
    scoremaps = apply_gaussian_kernel(scoremaps.view(B, Nq, h2, w2), sigma).view(B, Nq, -1)
    scoremaps = softmax_with_temperature(scoremaps, softmax_temp, -1)

    grid = create_grid(h2, w2, device=_device)

    # average over grid
    scoremaps = scoremaps.unsqueeze(-1).expand(-1, -1, -1, 2)
    grid = grid.view(-1, 2).unsqueeze(0).unsqueeze(0).expand(B, Nq, -1, -1)
    matches = (scoremaps * grid).sum(dim=2)

    return matches


def apply_gaussian_kernel(scoremaps, sigma=7):
    '''
    Apply a gaussian kernel centered at the maximum point
    IN:
        scoremaps [torch.Tensor] (B x Nq x h x w)
    OUT:
        Supressed scoremaps [torch.Tensor] (B x Nq x h x w)
    '''
    B, Nq, h, w = scoremaps.shape
    _device = scoremaps.device

    idx = torch.max(scoremaps.view(B, Nq, -1), dim=-1).indices
    idx_y = (idx // w).view(B, Nq, 1, 1).float()
    idx_x = (idx % w).view(B, Nq, 1, 1).float()
    
    grid = create_grid(h, w, device=_device).unsqueeze(0).unsqueeze(0)
    grid = grid.expand(B, Nq, -1, -1, -1)
    x = grid[..., 0]
    y = grid[..., 1]

    gauss_kernel = torch.exp(-((x-idx_x)**2 + (y-idx_y)**2) / (2 * sigma**2))

    return gauss_kernel * scoremaps