"""
* @auther tozawa
* @date 2018-7-17
*
* Evaluate each evaluation criteria
"""
import os, sys, time
import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_mse as mse
from scipy.fftpack import fftn

def calc_zncc(im1, im2):
    """
    https://github.com/ladisk/pyDIC/blob/master/dic.py
    Calculate the zero normalized cross-correlation coefficient of input images.
    :param im1: First input image.
    :param im2: Second input image.
    :return: zncc ([0,1]). If 1, input images match perfectly.
    """
    nom = np.mean((im1-im1.mean())*(im2-im2.mean()))
    den = im1.std()*im2.std()
    if den == 0:
        return 0
    return nom/den

def calc_score_on_fft_domain(gt, input):
    """
    Calculate mean absolute error in power spectrum and MAE in phase
    This score is original. I assume arguments are followed as:
    @param input: input image
    @param gt: ground truth
    return: diff_spe_const, diff_spe, diff_ang_const, diff_ang
    """
    # FFT
    imgFreqs = fftn(input)
    gtFreqs = fftn(gt)
    imgFreqs = np.fft.fftshift(imgFreqs)
    gtFreqs = np.fft.fftshift(gtFreqs)

    # Difference of power spectrum
    diff_spe = np.absolute((np.abs(gtFreqs) ** 2)-(np.abs(imgFreqs) ** 2))
    diff_spe_const = np.mean(diff_spe)

    # Difference of angle
    diff_ang = np.abs(np.angle(imgFreqs/gtFreqs, deg=True))
    diff_ang_const = np.mean(diff_ang)

    return diff_spe_const, diff_spe, diff_ang_const, diff_ang

def calc_mse(gt, img):
    """
    Calculate mean squared error
    @param gt: ground truth
    @param: img: input image
    """
    mse_const = mse(gt, img)
    return mse_const

def calc_psnr(gt, img):
    """
    Calculate peak to signal noise ratio
    @param gt: ground truth
    @param: img: input image
    """
    psnr_const = psnr(gt, img, data_range =255)
    return psnr_const

def calc_ssim(gt, img):
    """
    Calculate structural similarity index
    @param gt: ground truth
    @param: img: input image
    """
    ssim_const = ssim(gt, img, data_range =255, gaussian_weights=True, use_sample_covariance=False)
    return ssim_const

def calc_dice(a, b):
    """
    https://github.com/simizlab/B4Contest/blob/master/example/evaluation.py#L40
    Args:
        a: bool image
        b: bool image
    """
    # Compute Dice coefficient
    return 2. * (a & b).sum() / (a.sum() + b.sum())

def calc_jaccard_index(im1, im2):
    """
    https://github.com/zEttOn86/3D-Unet/blob/master/evaluations/calc_ji.py#L7
    """
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)

    union = np.logical_or(im1, im2)

    if float(intersection.sum()) == 0.:
        return 0.
    else:
        return intersection.sum() / float(union.sum())
