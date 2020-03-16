#coding: utf-8
"""
* @auther ryosuke
* For completely unpairSR
"""

"""
Train 
org 1240*1299*3600
mask 1240*1210*3600

train1 for LR  1240*0≦y≦799*3600  //41529patches
train2 for HR  1240*800≦y≦1210*3600 //48994patches

input  train.mhd
output train1.mhd  train2.mhd
"""

"""
めも

・mhdロード
↓
・分割
↓
・保存

"""
#coding:utf-8
import os, sys, time
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
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

    file_name='test/denoising/MicroCT.mhd'
    file_name2 = 'validation/denoising/MicroCT.mhd'
    # file_name='train/denoising/train1/downsampling/Tri/x8/train1.mhd'
    # file_name2 = 'train/denoising/train2/calc_mask2.mhd'
    # file_name = 'train/denoising/train2/train2.mhd'
    #file_name = 'train/denoising/MicroCT.mhd'
    # file_name2 = 'train/calc_mask.mhd'

    #make output dir
    result_dir = os.path.join(args.base, args.out)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    #load image data
    print("train.mhd load")
    sitktrain = sitk.ReadImage(os.path.join(args.root,file_name))
    train1 = sitk.GetArrayFromImage(sitktrain).astype("float32")#たしかこれ正規化済
    print("train.mhd load done")

    print("train.mhd load")
    sitktrain = sitk.ReadImage(os.path.join(args.root,file_name2))
    train2 = sitk.GetArrayFromImage(sitktrain).astype("float32")#たしかこれ正規化済
    print("train.mhd load done")

    max1=train1.max()
    min1 = train1.min()
    max2=train2.max()
    min2 = train2.min()

    print(max1,"\\",min1)
    print(max2, "\\", min2)

    # print("mask.mhd load")
    # sitkmask = sitk.ReadImage(os.path.join(args.root,file_name2))
    # mask = sitk.GetArrayFromImage(sitkmask).astype("float32")#たしかこれ正規化済
    # print("mask.mhd load done")

    #train cut
    #train1_1=train1[2500:2610,700:710,744:994]
    #train2_1 = train2[2600:2710, 10:20,744:994]
    # train1_2 = train1[2500:2610, 615:620,650:900]
    #train2_2 = train2[3000:3110, 170:180,540:790]
    # mask1=mask[:,0:800,:]
    # mask2 = mask[:, 800:1210, :]
    # train1=train[:,0:1296,:]
    #
    # mask1=mask[:,0:1296,:]

    # print(train1.shape)
    # print(mask1.shape)


    # train11,train22 =np.split(train,[1210],1)
    # mask11,mask22 =np.split(mask,[1210],1)
    # print(train11.shape)
    # print(mask11.shape)
    # print(train2.shape)

    # train1=train[:,0:1210,:]
    # mask1 = mask[:, 0:1210, :]

    #save images

    # print("images save")
    # train1_1 = train1_1.flatten()
    # train1_2 = train1_2.flatten()
    # train2_1 = train2_1.flatten()
    # train2_2 = train2_2.flatten()
    # mask1 = mask1.flatten()
    #train2 = train2.flatten()
    # mask2 = mask2.flatten()


    # IO.write_mhd_and_raw_withoutSitk(train1_1, result_dir + '/train1_1.mhd',
    #                                  ndims=3, size=[250,10,110],
    #                                  space=[0.070, 0.066, 0.070])
    # IO.write_mhd_and_raw_withoutSitk(train1_2, result_dir + '/train1_2.mhd',
    #                                  ndims=3, size=[250,10,110],
    #                                  space=[0.070, 0.066, 0.070])
    # IO.write_mhd_and_raw_withoutSitk(train2_1, result_dir + '/train2_1.mhd',
    #                                  ndims=3, size=[250,10,110],
    #                                  space=[0.070, 0.066, 0.070])
    # IO.write_mhd_and_raw_withoutSitk(train2_2, result_dir + '/train2_2.mhd',
    #                                  ndims=3, size=[250,10,110],
    #                                  space=[0.070, 0.066, 0.070])
    # IO.write_mhd_and_raw_withoutSitk(train1, result_dir + '/train1.mhd',
    #                                  ndims=3, size=[1240,800,3600],
    #                                  space=[0.070, 0.066, 0.070])


    # IO.write_mhd_and_raw_withoutSitk(mask1, result_dir + '/calc_mask1.mhd',
    #                                      ndims=3, size=[1240,800,3600],
    #                                      space=[0.070, 0.066, 0.070])
    # IO.write_mhd_and_raw_withoutSitk(mask2, result_dir + '/calc_mask2.mhd',
    #                                      ndims=3, size=[1240,410,3600],
    #                                      space=[0.070, 0.066, 0.070])
    # IO.write_mhd_and_raw_withoutSitk(train1, result_dir + '/MicroCT.mhd',
    #                                  ndims=3, size=[1240,1296,3600],
    #                                  space=[0.070, 0.066, 0.070])
    # IO.write_mhd_and_raw_withoutSitk(mask1, result_dir + '/calc_mask.mhd',
    #                                      ndims=3, size=[1240,1296,3600],
    #                                      space=[0.070, 0.066, 0.070])

if __name__ == '__main__':
    main()



