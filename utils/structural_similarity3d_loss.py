#coding:utf-8
import chainer
import chainer.functions as F
import numpy as np
from math import exp

def gaussian(window_size, sigma, xp):
    """
    https://daily.belltail.jp/?p=2457
    """
    x = xp.arange(0, window_size, dtype=xp.float32)
    gauss = xp.exp(-(x-window_size//2)**2/(2*sigma**2))
    # gauss = chainer.Variable(xp.array([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)], dtype=xp.float32))
    return gauss/xp.sum(gauss)

def create_window(window_size, channel, xp):
    weight = gaussian(window_size, 1.5, xp)

    x_window = weight.reshape(1, 1, 1, 1, -1) # (out_ch, in_ch, z, y, x)
    x_window = xp.repeat(x_window, channel, axis=0)

    y_window = weight.reshape(1, 1, 1, -1, 1) # (out_ch, in_ch, z, y, x)
    y_window = xp.repeat(y_window, channel, axis=0)

    z_window = weight.reshape(1, 1, -1, 1, 1) # (out_ch, in_ch, z, y, x)
    z_window = xp.repeat(z_window, channel, axis=0)

    return x_window, y_window, z_window

def gaussian_filter(img, window, pad, channel):
    x_window, y_window, z_window = window
    h = F.convolution_3d(img, x_window, pad=(0, 0, pad), groups=channel)
    h = F.convolution_3d(h, y_window, pad=(0, pad, 0), groups=channel)
    return F.convolution_3d(h, z_window, pad=(pad, 0, 0), groups=channel)

def _calc_ssim_map(img1, img2, window, window_size, channel, data_range):
    mu1 = gaussian_filter(img1, window, pad=window_size//2, channel=channel)
    mu2 = gaussian_filter(img2, window, pad=window_size//2, channel=channel)

    mu1_sq = F.square(mu1)
    mu2_sq = F.square(mu2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(img1*img1, window, pad=window_size//2, channel=channel) - mu1_sq
    sigma2_sq = gaussian_filter(img2*img2, window, pad=window_size//2, channel=channel) - mu2_sq
    sigma12 = gaussian_filter(img1*img2, window, pad=window_size//2, channel=channel) - mu1_mu2

    C1 = (0.01*data_range)**2
    C2 = (0.03*data_range)**2

    luminance = (2*mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    cs = (2*sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)

    ssim_map = luminance * cs

    return ssim_map

def structural_dissimilarity3d_loss(img1, img2, window_size = 11, data_range=None):
    if data_range is None:
        data_range = F.max(img1) - F.min(img1)
    (_, channel, _, _, _) = img1.shape
    xp = chainer.backends.cuda.get_array_module(img1)
    window = create_window(window_size, channel, xp)
    ssim_map = _calc_ssim_map(img1, img2, window, window_size, channel, data_range)
    return F.mean((1. - ssim_map))
