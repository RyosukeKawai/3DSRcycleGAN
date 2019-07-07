#coding:utf-8
import os, sys, time
import argparse
import numpy as np
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '..')))
import util.ioFunction_version_4_3 as IO

def zscore(x, axis = None):
    """
    https://deepage.net/features/numpy-normalize.html
    """
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean)/xstd
    return zscore

def min_max(x, axis=None):
    """
    https://deepage.net/features/numpy-normalize.html
    """
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

def main():
    parser = argparse.ArgumentParser(description='Make normalized image')
    parser.add_argument('--inputImageFile', '-i', help='input image file')
    parser.add_argument('--outputImageFile', '-o', help='output image file')
    args = parser.parse_args()

    input_image, dict = IO.read_mhd_and_raw_withoutSitk(args.inputImageFile)

    normed_img = zscore(input_image)

    IO.write_mhd_and_raw_withoutSitk(normed_img,
                                        args.outputImageFile,
                                        ndims=3,
                                        size=dict['DimSize'],
                                        space=dict['ElementSpacing'])

if __name__ == '__main__':
    main()
