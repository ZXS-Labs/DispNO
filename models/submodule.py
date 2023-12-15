from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.autograd.function import Function
import torch.nn.functional as F
import numpy as np
import math


def convbn(in_channels, out_channels, kernel_size, stride, pad, dilation):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False),
                         nn.BatchNorm2d(out_channels))

def convbn_3d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                   padding=pad, bias=False),
                         nn.BatchNorm3d(out_channels))

def scale_coords(points, max_length):
    return torch.clamp(2 * points/(max_length-1.)- 1., -1., 1.)

def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.view(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=False)

def build_concat_dynamic_interp_volume(refimg_fea, targetimg_fea, maxdisp, scale):
    B, C, H, W = refimg_fea.shape
    step = 192 / (maxdisp * scale)
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for s in range(B):
        for i in range(maxdisp):
            if i > 0:
                nowStep = (i * step[s]).cpu().numpy()
                startIdx = math.ceil(nowStep)
                ref_nu = np.linspace(startIdx, W - 1, W - startIdx)
                tar_nu = ref_nu - nowStep
                nv = np.linspace(0, H - 1, H)
                tar_u, tar_v = np.meshgrid(tar_nu, nv)
                tar_u = tar_u.flatten()
                tar_v = tar_v.flatten()
                coord = np.stack((tar_u, tar_v), axis=-1)
                coord = torch.tensor(coord).float()
                coord = coord.permute(1, 0).unsqueeze(0)
                u = scale_coords(coord[:, 0:1, :], W)
                v = scale_coords(coord[:, 1:2, :], H)
                coord = torch.cat([u,v],1).cuda()
                tar_fea = targetimg_fea[s, :, :, :].unsqueeze(0)
                interp_feat = interpolate(tar_fea, coord)

                volume[s, :C, i, :, startIdx:] = refimg_fea[s, :, :, startIdx:]
                volume[s, C:, i, :, startIdx:] = interp_feat.reshape(1, C, H, -1).squeeze(0)
            else:
                volume[s, :C, i, :, :] = refimg_fea[s, :, :, :]
                volume[s, C:, i, :, :] = targetimg_fea[s, :, :, :]
    volume = volume.contiguous()
    return volume

def build_concat_dynamic_interp_volume_range(refimg_fea, targetimg_fea, maxdisp, scale, start_disp, end_disp):
    B, C, H, W = refimg_fea.shape
    step = (end_disp - start_disp) / (maxdisp * scale)
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    startStep = start_disp / scale
    for s in range(B):
        for i in range(maxdisp):
            nowStep = startStep[s].cpu().numpy() + (i * step[s]).cpu().numpy()
            startIdx = math.ceil(nowStep)
            ref_nu = np.linspace(startIdx, W - 1, W - startIdx)
            tar_nu = ref_nu - nowStep
            nv = np.linspace(0, H - 1, H)
            tar_u, tar_v = np.meshgrid(tar_nu, nv)
            tar_u = tar_u.flatten()
            tar_v = tar_v.flatten()
            coord = np.stack((tar_u, tar_v), axis=-1)
            coord = torch.tensor(coord).float()
            coord = coord.permute(1, 0).unsqueeze(0)
            u = scale_coords(coord[:, 0:1, :], W)
            v = scale_coords(coord[:, 1:2, :], H)
            coord = torch.cat([u,v],1).cuda()
            tar_fea = targetimg_fea[s, :, :, :].unsqueeze(0)
            interp_feat = interpolate(tar_fea, coord)

            volume[s, :C, i, :, startIdx:] = refimg_fea[s, :, :, startIdx:]
            volume[s, C:, i, :, startIdx:] = interp_feat.reshape(1, C, H, -1).squeeze(0)
    volume = volume.contiguous()
    return volume

def build_concat_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
            volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
        else:
            volume[:, :C, i, :, :] = refimg_fea
            volume[:, C:, i, :, :] = targetimg_fea
    volume = volume.contiguous()
    return volume

def groupwise_correlation_dynamic(fea1, fea2, num_groups):
    # num_groups = 40
    C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([num_groups, channels_per_group, H, W]).mean(dim=1)
    assert cost.shape == (num_groups, H, W)
    return cost

def groupwise_correlation(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W]).mean(dim=2)
    assert cost.shape == (B, num_groups, H, W)
    return cost

def groupwise_kl(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups
    fea1Group = fea1.view([B, num_groups, channels_per_group, H, W])
    fea2Group = fea2.view([B, num_groups, channels_per_group, H, W])
    log_fea1Group = torch.log(fea1Group) + 0.000000001
    log_fea2Group = torch.log(fea2Group) + 0.000000001
    kl_fea = fea1Group * (log_fea1Group - log_fea2Group)
    cost =torch.sum(kl_fea, dim=2)
    cost = cost / channels_per_group
    assert cost.shape == (B, num_groups, H, W)
    return cost

def groupwise_cosine(fea1, fea2, num_groups):

    B, C, H, W = fea1.shape

    assert C % num_groups == 0

    channels_per_group = C // num_groups

    fea1Group = fea1.view([B, num_groups, channels_per_group, H, W])

    fea2Group = fea2.view([B, num_groups, channels_per_group, H, W])

    cost = F.cosine_similarity(fea1Group, fea2Group, dim=2)

    assert cost.shape == (B, num_groups, H, W)

    return cost
def groupwise_cosine_manual(fea1, fea2, num_groups):
    B, C, H, W = fea1.shape
    assert C % num_groups == 0
    channels_per_group = C // num_groups    
    cost = (fea1 * fea2).view([B, num_groups, channels_per_group, H, W])
    cost = torch.sum(cost, dim=2)
    fea1_2 = torch.sum((fea1 * fea1).view([B, num_groups, channels_per_group, H, W]), dim=2)
    fea2_2 = torch.sum((fea2 * fea2).view([B, num_groups, channels_per_group, H, W]), dim=2) 
    fea1_ = torch.sqrt(fea1_2) + 0.00000001
    fea2_ = torch.sqrt(fea2_2) + 0.00000001

    ret_cost = cost / (fea1_ * fea2_)
    assert ret_cost.shape == (B, num_groups, H, W)
    return ret_cost
 
def variance(fea1, fea2):
    B, C, H, W = fea1.shape
    ave1 = (fea1 + fea2) / 2
    ave2 = (fea1.pow(2) + fea2.pow(2)) / 2
    cost = ave2 - ave1.pow(2)
    return cost

def build_gwc_piancha_dynamic_interp_volume(refimg_fea, targetimg_fea, maxdisp, num_groups, scale):
    # num_groups = 40
    B, C, H, W = refimg_fea.shape
    step = 192 / (maxdisp * scale)
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for s in range(B):
        for i in range(maxdisp):
            if i > 0:
                nowStep = (i * step[s]).cpu().numpy()
                startIdx = math.ceil(nowStep)
                ref_nu = np.linspace(startIdx, W - 1, W - startIdx)
                tar_nu = ref_nu - nowStep
                nv = np.linspace(0, H - 1, H)
                tar_u, tar_v = np.meshgrid(tar_nu, nv)
                tar_u = tar_u.flatten()
                tar_v = tar_v.flatten()
                coord = np.stack((tar_u, tar_v), axis=-1)
                coord = torch.tensor(coord).float()
                coord = coord.permute(1, 0).unsqueeze(0)
                u = scale_coords(coord[:, 0:1, :], W)
                v = scale_coords(coord[:, 1:2, :], H)
                coord = torch.cat([u,v],1).cuda()
                tar_fea = targetimg_fea[s, :, :, :].unsqueeze(0)
                interp_feat = interpolate(tar_fea, coord)
                ref = refimg_fea[s, :, :, startIdx:]
                tar = interp_feat.reshape(1, C, H, -1).squeeze(0)
                avg = (ref + tar) / 2
                ref = ref - avg
                tar = tar - avg
                volume[s, :, i, :, startIdx:] = groupwise_correlation_dynamic(ref,tar,
                                                            num_groups)
            else:
                ref = refimg_fea[s, :, :, :]
                tar = targetimg_fea[s, :, :, :]
                avg = (ref + tar) / 2
                ref = ref - avg
                tar = tar - avg
                volume[s, :, i, :, :] = groupwise_correlation_dynamic(ref, tar, num_groups)
    volume = volume.contiguous()
    return volume

def build_gwc_piancha_dynamic_interp_volume_range(refimg_fea, targetimg_fea, maxdisp, num_groups, scale, start_disp, end_disp):
    # num_groups = 40
    B, C, H, W = refimg_fea.shape
    step = (end_disp - start_disp) / (maxdisp * scale)
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    startStep = start_disp / scale
    for s in range(B):
        for i in range(maxdisp):
            nowStep = startStep[s].cpu().numpy() + (i * step[s]).cpu().numpy()
            startIdx = math.ceil(nowStep)
            ref_nu = np.linspace(startIdx, W - 1, W - startIdx)
            tar_nu = ref_nu - nowStep
            nv = np.linspace(0, H - 1, H)
            tar_u, tar_v = np.meshgrid(tar_nu, nv)
            tar_u = tar_u.flatten()
            tar_v = tar_v.flatten()
            coord = np.stack((tar_u, tar_v), axis=-1)
            coord = torch.tensor(coord).float()
            coord = coord.permute(1, 0).unsqueeze(0)
            u = scale_coords(coord[:, 0:1, :], W)
            v = scale_coords(coord[:, 1:2, :], H)
            coord = torch.cat([u,v],1).cuda()
            tar_fea = targetimg_fea[s, :, :, :].unsqueeze(0)
            interp_feat = interpolate(tar_fea, coord)
            ref = refimg_fea[s, :, :, startIdx:]
            tar = interp_feat.reshape(1, C, H, -1).squeeze(0)
            avg = (ref + tar) / 2
            ref = ref - avg
            tar = tar - avg
            volume[s, :, i, :, startIdx:] = groupwise_correlation_dynamic(ref,tar,
                                                        num_groups)
    volume = volume.contiguous()
    return volume

def build_gwc_piancha_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            ref = refimg_fea[:, :, :, i:]
            tar = targetimg_fea[:, :, :, :-i]
            avg = (ref + tar) / 2
            avg2 = (ref.pow(2) + tar.pow(2)) / 2
            ref = ref - avg
            tar = tar - avg
            volume[:, :, i, :, i:] = groupwise_correlation(ref,tar,
                                                           num_groups)
        else:
            ref = refimg_fea
            tar = targetimg_fea
            avg = (ref + tar) / 2
            ref = ref - avg
            tar = tar - avg
            volume[:, :, i, :, :] = groupwise_correlation(ref, tar, num_groups)
    volume = volume.contiguous()
    return volume

def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_correlation(refimg_fea[:, :, :, i:],
                                                           targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_correlation(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume

def build_gwcosine_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_cosine(refimg_fea[:, :, :, i:],
                                                           targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_cosine(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume
def build_gwcosine_manual_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_cosine_manual(refimg_fea[:, :, :, i:],
               targetimg_fea[:, :, :, :-i],
                  num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_cosine_manual(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume

def build_gwkl_volume(refimg_fea, targetimg_fea, maxdisp, num_groups):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = groupwise_kl(refimg_fea[:, :, :, i:],
                                                           targetimg_fea[:, :, :, :-i],
                                                           num_groups)
        else:
            volume[:, :, i, :, :] = groupwise_kl(refimg_fea, targetimg_fea, num_groups)
    volume = volume.contiguous()
    return volume

def build_variance_volume(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape
    volume = refimg_fea.new_zeros([B, C, maxdisp, H, W])
    for i in range(maxdisp):
        if i > 0:
            volume[:, :, i, :, i:] = variance(refimg_fea[:, :, :, i:],
                                                           targetimg_fea[:, :, :, :-i])                                                          
        else:
            volume[:, :, i, :, :] = variance(refimg_fea, targetimg_fea)
    volume = volume.contiguous()
    return volume

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out
    
def interpolate(feat, uv):
    uv = uv.transpose(1, 2)
    uv = uv.unsqueeze(2)
    samples = torch.nn.functional.grid_sample(feat, uv)
    return samples[:, :, :, 0]


