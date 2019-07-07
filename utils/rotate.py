#coding:utf-8
import os, sys, time
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F

def rotate90(images):
    if not images.ndim == 5:
        raise NotImplementedError()
    return images.transpose(0, 1, 2, 4, 3)[:,:,:,:,::-1]

def rotate180(images):
    if not images.ndim == 5:
        raise NotImplementedError()
    return images[:,:,:,::-1,::-1]

def rotate270(images):
    if not images.ndim == 5:
        raise NotImplementedError()
    return images.transpose(0, 1, 2, 4, 3)[:,:,:,::-1,:]

def rotate_images(images, rot90_scalars=(0, 1, 2, 3)):
    """
    Return the input image and its 90, 180, and 270 degree rotations along z axis.
    https://github.com/google/compare_gan/blob/master/compare_gan/gans/utils.py
    """
    images_rotated = [
        images,
        rotate90(images),
        rotate180(images),
        rotate270(images)
    ]
    results = F.stack([images_rotated[i] for i in rot90_scalars])
    results = F.reshape(results, [-1] + list(images.shape[1:]))
    return results
