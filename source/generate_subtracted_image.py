#coding:utf-8
import os, sys, time
import argparse
import numpy as np
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '..')))
import util.ioFunction_version_4_3 as IO

def main():
    parser = argparse.ArgumentParser(description='Make subtracted image')
    parser.add_argument('--inputImageFile1', '-i1', help='input image file')
    parser.add_argument('--inputImageFile2', '-i2', help='input image file')
    parser.add_argument('--outputImageFile', '-o', help='output image file')
    args = parser.parse_args()

    img1, dict = IO.read_mhd_and_raw_withoutSitk(args.inputImageFile1)
    img2, _ = IO.read_mhd_and_raw_withoutSitk(args.inputImageFile2)

    out = (img1-img2)**2
    print('MSE = {}'.format(np.mean(out)))
    IO.write_mhd_and_raw_withoutSitk(out,
                                        args.outputImageFile,
                                        ndims=3,
                                        size=dict['DimSize'],
                                        space=dict['ElementSpacing'])

if __name__ == '__main__':
    main()
