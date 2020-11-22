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
    psnr_const = psnr(gt, img, dynamic_range =255)
    return psnr_const

def calc_ssim(gt, img):
    """
    Calculate structural similarity index
    @param gt: ground truth
    @param: img: input image
    """
    ssim_const = ssim(gt, img, dynamic_range =255, gaussian_weights=True, use_sample_covariance=False)
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

def calc_dice(im1, im2):
    # im1 = np.asarray(im1).astype(np.bool)
    # im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    if float(intersection.sum()) == 0.:
        return 0.
    else:
        return 2. * intersection.sum() / (im1.sum() + im2.sum())

def calc_alveoliness(im1, im2, min_cycle=5, max_cycle=31, save_flag=False):
    """
    Args:
        im1: img (numpy ndarray)
        im2: img (numpy ndarray)
        min_cycle: 
                minimum cycle i.e. minimum period of alveoli
                default is 5. This number is decided experimentary.
        max_cycle: 
                maximum cycle i.e. maximum period of alveoli.
                default is 31. This number equals a minimum period of alveoli. 
        save_flag:
                Do you wanna save img? if True, this function return [const alveoli, band_absolute_error]
    """

    low_filtering_freq = [int(i/max_cycle+0.5) for i in im1.shape]
    high_filtering_freq = [int(i/min_cycle+0.5) for i in im1.shape]

    im1_freq = apply_fft(im1)
    im2_freq = apply_fft(im2)

    im1_power_spe = np.abs(im1_freq)**2
    im2_power_spe = np.abs(im2_freq)**2

    absolute_error = np.abs((im1_power_spe - im2_power_spe))

    band_absolute_error = get_band_power_spectrum(absolute_error, 
                                                        high_filtering_freq, 
                                                        low_filtering_freq)

    N = im1_freq.size
    Hz, Hy, Hx = high_filtering_freq
    Lz, Ly, Lx = low_filtering_freq

    const_band_error = np.sum(band_absolute_error)/(8*(Hz*Hy*Hx-Lz*Ly*Lx))
    if not save_flag:
        return const_band_error

    return const_band_error, band_absolute_error


def apply_fft(img):
    temp_freq = fftn(img)
    freq = np.fft.fftshift(temp_freq)
    return freq

def apply_ifft(freq):
    temp_freq = np.fft.ifftshift(freq)
    img = ifftn(temp_freq)
    return img.real

def apply_low_pass_filter(freq, filtering_freq):
    """
    Low pass filter
    F_l >= F can pass
    """
    zc, yc, xc = [ i//2 for i in freq.shape]
    z_filtering_freq, y_filtering_freq, x_filtering_freq = filtering_freq
    zs, ze = zc - z_filtering_freq, zc + z_filtering_freq
    ys, ye = yc - y_filtering_freq, yc + y_filtering_freq
    xs, xe = xc - x_filtering_freq, xc + x_filtering_freq

    temp_freq = np.zeros_like(freq)
    temp_freq[zs:ze, ys:ye, xs:xe] = freq[zs:ze, ys:ye, xs:xe]

    return temp_freq

def apply_high_pass_filter(freq, filtering_freq):
    """
    High pass filter
    F_h <= F can pass
    """
    zc, yc, xc = [ i//2 for i in freq.shape]
    z_filtering_freq, y_filtering_freq, x_filtering_freq = [i-1 for i in filtering_freq]
    zs, ze = zc - z_filtering_freq, zc + z_filtering_freq
    ys, ye = yc - y_filtering_freq, yc + y_filtering_freq
    xs, xe = xc - x_filtering_freq, xc + x_filtering_freq

    temp_freq = freq.copy()
    temp_freq[zs:ze, ys:ye, xs:xe] = 0 + 0j

    return temp_freq

def apply_band_pass_filter(freq, high_filtering_freq, low_filtering_freq):
    """
    Band pass filter
    F_h >= F >= F_l can pass
    """
    zc, yc, xc = [ i//2 for i in freq.shape]
    z_high_freq, y_high_freq, x_high_freq = high_filtering_freq
    z_low_freq, y_low_freq, x_low_freq = [i-1 for i in low_filtering_freq]

    hz_s, hz_e = zc - z_high_freq, zc + z_high_freq
    hy_s, hy_e = yc - y_high_freq, yc + y_high_freq
    hx_s, hx_e = xc - x_high_freq, xc + x_high_freq

    lz_s, lz_e = zc - z_low_freq, zc + z_low_freq
    ly_s, ly_e = yc - y_low_freq, yc + y_low_freq
    lx_s, lx_e = xc - x_low_freq, xc + x_low_freq

    temp_freq = np.zeros_like(freq)
    temp_freq[hz_s:hz_e, hy_s:hy_e, hx_s:hx_e] = freq[hz_s:hz_e, hy_s:hy_e, hx_s:hx_e]
    temp_freq[lz_s:lz_e, ly_s:ly_e, lx_s:lx_e] = 0 + 0j

    return temp_freq

def get_low_power_spectrum(power_spectrum, filtering_freq):
    zc, yc, xc = [ i//2 for i in power_spectrum.shape]
    z_filtering_freq, y_filtering_freq, x_filtering_freq = filtering_freq
    zs, ze = zc - z_filtering_freq, zc + z_filtering_freq
    ys, ye = yc - y_filtering_freq, yc + y_filtering_freq
    xs, xe = xc - x_filtering_freq, xc + x_filtering_freq

    temp_freq = np.zeros_like(power_spectrum)
    temp_freq[zs:ze, ys:ye, xs:xe] = power_spectrum[zs:ze, ys:ye, xs:xe]

    return temp_freq

def get_high_power_spectrum(power_spectrum, filtering_freq):
    zc, yc, xc = [ i//2 for i in power_spectrum.shape]
    z_filtering_freq, y_filtering_freq, x_filtering_freq = [i-1 for i in filtering_freq]
    zs, ze = zc - z_filtering_freq, zc + z_filtering_freq
    ys, ye = yc - y_filtering_freq, yc + y_filtering_freq
    xs, xe = xc - x_filtering_freq, xc + x_filtering_freq
    
    temp_freq = power_spectrum.copy()
    temp_freq[zs:ze, ys:ye, xs:xe] = 0

    return temp_freq

def get_band_power_spectrum(power_spectrum, high_filtering_freq, low_filtering_freq):
    """
    Band pass filter
    F_h >= F >= F_l can pass
    """
    zc, yc, xc = [ i//2 for i in power_spectrum.shape]
    z_high_freq, y_high_freq, x_high_freq = high_filtering_freq
    z_low_freq, y_low_freq, x_low_freq = [i-1 for i in low_filtering_freq]

    hz_s, hz_e = zc - z_high_freq, zc + z_high_freq
    hy_s, hy_e = yc - y_high_freq, yc + y_high_freq
    hx_s, hx_e = xc - x_high_freq, xc + x_high_freq

    lz_s, lz_e = zc - z_low_freq, zc + z_low_freq
    ly_s, ly_e = yc - y_low_freq, yc + y_low_freq
    lx_s, lx_e = xc - x_low_freq, xc + x_low_freq

    temp_freq = np.zeros_like(power_spectrum)
    temp_freq[hz_s:hz_e, hy_s:hy_e, hx_s:hx_e] = power_spectrum[hz_s:hz_e, hy_s:hy_e, hx_s:hx_e]
    temp_freq[lz_s:lz_e, ly_s:ly_e, lx_s:lx_e] = 0

    return temp_freq