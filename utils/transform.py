#coding:utf-8
import os, sys, time
import numpy as np
sys.path.append(os.path.dirname(__file__))
import ioFunctions as IO

def transform(in_data):
    img1, img2 = in_data

    # Select surface
    # Let selected surface be on top
    surface = np.random.randint(0, 6)
    img1 = top_of_the_world(img1, surface)
    img2 = top_of_the_world(img2, surface)

    # rotation
    rot90_scalar = np.random.randint(0, 4)
    img1 = rotate_img(img1, rot90_scalar, axis='z')
    img2 = rotate_img(img2, rot90_scalar, axis='z')

    # For mirror image
    if np.random.rand() > 0.5:
        img1 = img1[:, :, :, ::-1]
        img2 = img2[:, :, :, ::-1]

    return img1, img2

#############################
# utils
#############################

def rotate90(img, axis='z'):
    """
    Args:
        img: shape is (ch, D, H, W)
    """
    if axis.lower() == 'x':
        return img.transpose(0, 2, 1, 3)[:,::-1,:,:]
    elif axis.lower() == 'y':
        return img.transpose(0, 3, 2, 1)[:,:,:,::-1]
    elif axis.lower() == 'z':
        return img.transpose(0, 1, 3, 2)[:,:,:,::-1]
    else:
        raise NotImplementedError()

def rotate180(img, axis='z'):
    """
    Args:
        img: shape is (ch, D, H, W)
    """
    if axis.lower() == 'x':
        return img[:, ::-1, ::-1, :]
    elif axis.lower() == 'y':
        return img[:, ::-1, :, ::-1]
    elif axis.lower() == 'z':
        return img[:, :, ::-1, ::-1]
    else:
        raise NotImplementedError()

def rotate270(img, axis='z'):
    """
    Args:
        img: shape is (ch, D, H, W)
    """
    if axis.lower() == 'x':
        return img.transpose(0, 2, 1, 3)[:, :, ::-1, :]
    elif axis.lower() == 'y':
        return img.transpose(0, 3, 2, 1)[:, ::-1, :, :]
    elif axis.lower() == 'z':
        return img.transpose(0, 1, 3, 2)[:, :, ::-1, :]
    else:
        raise NotImplementedError()

def rotate_img(img, rot90_scalar=0, axis='z'):
    """
    Return the input image and its 90, 180, and 270 degree rotations along z axis.
    https://github.com/google/compare_gan/blob/master/compare_gan/gans/utils.py
    """
    img_rotated = [
        img,
        rotate90(img, axis),
        rotate180(img, axis),
        rotate270(img, axis)
    ]
    return img_rotated[rot90_scalar]

def top_of_the_world(img, surface_selector=0):
    img_rotated = [
        img,                                            # surface 0
        rotate90(img, axis='x'),                        # surface 1
        rotate270(img, axis='y'),                       # surface 2
        rotate270(img, axis='x'),                       # surface 3
        rotate90(img, axis='y'),                        # surface 4
        rotate180(img, axis='x')                        # surface 5
    ]

    return img_rotated[surface_selector]
