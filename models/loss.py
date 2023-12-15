import torch
import torch.nn.functional as F


def model_loss(disp_ests, disp_gt, mask):
    weights = [0.5, 0.5, 0.7, 1.0]
    all_losses = []
    for disp_est, weight in zip(disp_ests, weights):
        all_losses.append(weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True))
    return sum(all_losses)

def gaussian(mu, sigma, labels):
    return torch.exp(-0.5*(mu-labels)** 2/ sigma** 2)/sigma

def laplacian(mu, b, labels):
    return 0.5 * torch.exp(-(torch.abs(mu-labels)/b))/b

def distribution(mu, sigma, labels, dist="gaussian"):
    return gaussian(mu, sigma, labels) if dist=="gaussian" else \
           laplacian(mu, sigma, labels)

def bimodal_loss(mu0, mu1, sigma0, sigma1, w0, w1, labels, dist="gaussian"):
    return - torch.log(w0 * distribution(mu0, sigma0, labels, dist) + \
                       w1 * distribution(mu1, sigma1, labels, dist))

def unimodal_loss(mu, sigma, labels):
    return torch.abs(mu - labels)/sigma + torch.log(sigma)

def smooth_l1_loss(preds, labels, reduce=None):
    return F.smooth_l1_loss(preds, labels, reduce=reduce)

def l1_loss(preds, labels, reduce=None):
    return F.l1_loss(preds, labels, reduce=reduce)