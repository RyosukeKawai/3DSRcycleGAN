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
import argparse, yaml, shutil, math
import numpy as np
import pandas as pd
import SimpleITK as sitk
import chainer

import util.ioFunction_version_4_3 as IO
import util.yaml_utils  as yaml_utils


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                        help='base directory path of program files')
    parser.add_argument('--out', '-o', default='results/',
                        help='Directory to output the result')
    parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Root directory path of input image')

    args = parser.parse_args()

    file_name='train/denoising/MicroCT.mhd'

    #make output dir
    result_dir = os.path.join(args.base, args.out)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    #load image data
    print("train.mhd load")
    sitktrain = sitk.ReadImage(os.path.join(args.root,file_name))
    train = sitk.GetArrayFromImage(sitktrain).astype("float32")#たしかこれ正規化済
    print("train.mhd load done")

    #train cut
    train1,train2 = np.split(train,[800],1)
    print(train1.shape)
    print(train2.shape)

    #save images

    print("images save")
    train1 = train1.flatten()
    train2 = train2.flatten()

    IO.write_mhd_and_raw_withoutSitk(train1, result_dir + '/train1.mhd',
                                     ndims=3, size=[1240,800,3600],
                                     space=[0.07, 0.066, 0.07])

    IO.write_mhd_and_raw_withoutSitk(train2, result_dir + '/train2.mhd',
                                     ndims=3, size=[1240,410,3600],
                                     space=[0.07, 0.066, 0.07])






