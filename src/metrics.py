import config
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from utils import recur


def SSIM(output, target, window_size=11, MAX=1.0, window=None, full=False):
    with torch.no_grad():
        MIN = 0
        if isinstance(output, torch.Tensor):
            output = output.expand(target.size())
            (_, channel, height, width) = output.size()
            if window is None:
                valid_size = min(window_size, height, width)
                sigma = 1.5
                gauss = torch.Tensor(
                    [math.exp(-(x - valid_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(valid_size)])
                gauss = gauss / gauss.sum()
                _1D_window = gauss.unsqueeze(1)
                _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
                window = _2D_window.expand(channel, 1, valid_size, valid_size).contiguous().to(config.PARAM['device'])
            mu1 = F.conv2d(output, window, padding=0, groups=channel)
            mu2 = F.conv2d(target, window, padding=0, groups=channel)
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            sigma1_sq = F.conv2d(output * output, window, padding=0, groups=channel) - mu1_sq
            sigma2_sq = F.conv2d(target * target, window, padding=0, groups=channel) - mu2_sq
            sigma12 = F.conv2d(output * target, window, padding=0, groups=channel) - mu1_mu2
            C1 = (0.01 * (MAX - MIN)) ** 2
            C2 = (0.03 * (MAX - MIN)) ** 2
            ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            ssim[torch.isnan(ssim)] = 1
            ssim = ssim.mean().item()
            if (full):
                cs = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
                cs[torch.isnan(cs)] = 1
                cs = cs.mean().item()
                return ssim, cs
    return ssim


def MSSIM(output, target, window_size=11, MAX=1.0):
    with torch.no_grad():
        weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=config.PARAM['device'])
        levels = weights.size()[0]
        if isinstance(output, torch.Tensor):
            mssim = []
            mcs = []
            for _ in range(levels):
                ssim, cs = SSIM(output, target, window_size=window_size, MAX=MAX, full=True)
                mssim.append(ssim)
                mcs.append(cs)
                output = F.avg_pool2d(output, 2)
                target = F.avg_pool2d(target, 2)
            mssim = torch.tensor(mssim, device=config.PARAM['device'])
            mcs = torch.tensor(mcs, device=config.PARAM['device'])
            pow1 = mcs ** weights
            pow2 = mssim ** weights
            mssim = torch.prod(pow1[:-1] * pow2[-1])
            mssim[torch.isnan(mssim)] = 0
            mssim = mssim.item()
        elif isinstance(output, list):
            mssim = 0
            for i in range(len(output)):
                mssim += MSSIM(output[i].unsqueeze(0), target[i].unsqueeze(0), window_size=window_size, MAX=MAX)
            mssim = mssim / len(output)
        else:
            raise ValueError('Data type not supported')
    return mssim


def PSNR(output, target, MAX=1.0):
    with torch.no_grad():
        max = torch.tensor(MAX).to(config.PARAM['device'])
        criterion = nn.MSELoss().to(config.PARAM['device'])
        mse = criterion(output, target)
        psnr = (20 * torch.log10(max) - 10 * torch.log10(mse)).item()
    return psnr


def BPP(code, img):
    with torch.no_grad():
        nbytes = code.nbytes
        num_pixels = img.numel() / img.size(1)
        bpp = 8 * nbytes / num_pixels
    return bpp


class Metric(object):
    def __init__(self):
        self.metric = {'Loss': (lambda input, output: output['loss'].item()),
                       'PSNR': (lambda input, output: recur(PSNR, output['img'], input['img'])),
                       'SSIM': (lambda input, output: recur(SSIM, output['img'], input['img'])),
                       'MSSIM': (lambda input, output: recur(MSSIM, output['img'], input['img'])),
                       'BPP': (lambda input, output: recur(BPP, output['code'], input['img']))}

    def evaluate(self, metric_names, input, output):
        evaluation = {}
        for metric_name in metric_names:
            evaluation[metric_name] = self.metric[metric_name](input, output)
        return evaluation