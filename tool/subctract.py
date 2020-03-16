#coding: utf-8
"""
* @auther ryosuke
* For completely unpairSR
"""
#coding:utf-8
import os, sys, time
import argparse
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as colors
import SimpleITK as sitk
import matplotlib.pyplot as plt
import chainer

import util.ioFunction_version_4_3 as IO



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                        help='base directory path of program files')
    parser.add_argument('--out', '-o', default='results/resize/',
                        help='Directory to output the result')
    parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Root directory path of input image')

    args = parser.parse_args()

    file_name='G:/data/test/denoising/MicroCT.mhd'
    file_name2 = 'G:/consideration_result_image/20190707/unpair_SR/inference/test/MicroCT.mhd'
    file_name3 = 'G:/consideration_result_image/20190707/unpair_SR-2/inference/test/MicroCT.mhd'
    file_name4 = 'G:/consideration_result_image/20190707/undergraduate_SR/inference/test/MicroCT.mhd'
    file_name5 = 'G:/consideration_result_image/20190707/undergraduate_SR-2/inference/MicroCT.mhd'

    sizex=640
    sizey=160
    sizez=480


    #make output dir
    result_dir = os.path.join(args.base, args.out)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    #load image data
    print(" load")
    sitktrain1 = sitk.ReadImage(os.path.join(file_name))
    img1 = sitk.GetArrayFromImage(sitktrain1).astype("float32")
    sitktrain2 = sitk.ReadImage(os.path.join(file_name2))
    img2 = sitk.GetArrayFromImage(sitktrain2).astype("float32")

    sitktrain3 = sitk.ReadImage(os.path.join(file_name3))
    img3 = sitk.GetArrayFromImage(sitktrain3).astype("float32")

    sitktrain4 = sitk.ReadImage(os.path.join(file_name4))
    img4 = sitk.GetArrayFromImage(sitktrain4).astype("float32")

    sitktrain5 = sitk.ReadImage(os.path.join(file_name5))
    img5 = sitk.GetArrayFromImage(sitktrain5).astype("float32")
    print("load done")

    #signed distance map
    fig = plt.figure()
    img_sub2 = fig.add_subplot(111, projection="3d")


    #subctract
    img_sub2 = (img1 - img2)+255.
    img_sub3 = (img1 - img3)+255.
    img_sub4 = (img1 - img4)+255.
    img_sub5 = (img1 - img5)+255.


    # #colors
    # norm = colors.Normalize(-255,255)
    # color_values = cm.gist_ncar(norm(img_sub2.flatten()))
    # plt.colorbar(img_sub2)
    #
    # #title
    # plt.title('undergraduate train_LR TOTAL 41529 patch')
    #
    # #plot
    # plt.show()

    # #save images
    #
    print("images save")
    img_sub2 = img_sub2.flatten()
    img_sub3 = img_sub3.flatten()
    img_sub4 = img_sub4.flatten()
    img_sub5 = img_sub5.flatten()
    #
    #
    #
    IO.write_mhd_and_raw_withoutSitk(img_sub2, result_dir + '/sub_A5_new.mhd',
                                     ndims=3, size=[sizex,sizey,sizez],
                                     space=[0.070, 0.066, 0.070])

    IO.write_mhd_and_raw_withoutSitk(img_sub3, result_dir + '/sub_A10_new.mhd',
                                     ndims=3, size=[sizex,sizey,sizez],
                                     space=[0.070, 0.066, 0.070])

    IO.write_mhd_and_raw_withoutSitk(img_sub4, result_dir + '/sub_A5_old.mhd',
                                     ndims=3, size=[sizex,sizey,sizez],
                                     space=[0.070, 0.066, 0.070])

    IO.write_mhd_and_raw_withoutSitk(img_sub5, result_dir + '/sub_A10_old.mhd',
                                     ndims=3, size=[sizex,sizey,sizez],
                                     space=[0.070, 0.066, 0.070])



if __name__ == '__main__':
    main()



