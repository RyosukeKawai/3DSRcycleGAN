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

input  denoised test.mhd
output test1-8.mhd  
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
    parser.add_argument('--out', '-o', default='results-jamit/',
                        help='Directory to output the result')
    parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Root directory path of input image')

    args = parser.parse_args()

    file_name='train/denoising/train1/train1.mhd'

    W = 250
    H = 1
    D = 110

    #make output dir
    result_dir = os.path.join(args.base, args.out)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    #load image data
    print("train1.mhd load")
    sitktrain = sitk.ReadImage(os.path.join(args.root,file_name))
    train = sitk.GetArrayFromImage(sitktrain).astype("float32")#たしかこれ正規化済
    print("train1.mhd load done")

    print(train.shape)

    count=0
    for i in range(0,1240-W):
        for j in range(0,800-H):
            for k in range(0,3600-D):
                check = train[k:k + D, j:j + H, i:i + W]
                std=np.std(check)
                if std>34.0:
                    count+=1
                    # print(count)
                    df=pd.DataFrame({'i':[i],'j':[j],'k':[k],'std':[std]})
                    df.to_csv('{}/results.csv'.format(result_dir), index=False, encoding='utf-8', mode='a')
                    check = check.flatten()
                    IO.write_mhd_and_raw_withoutSitk(check, result_dir + '/for_jamit/check{}.mhd'.format(count),
                                                      ndims=3, size=[250, 1, 110],
                                                      space=[0.07, 0.066, 0.07])

            # df=pd.DataFrame({'num':[j],'std':[std]})
            # df.to_csv('{}/results.csv'.format(result_dir),index=False, encoding='utf-8', mode='a')
            # print("num={} save".format(j))
            # test_patch=test_patch.flatten()
            # IO.write_mhd_and_raw_withoutSitk(test_patch, result_dir + '/for_jamit/test{}.mhd'.format(j),
            #                                  ndims=3, size=[250, 1, 110],
            #                                  space=[0.07, 0.066, 0.07])

    # debug
    # parser = argparse.ArgumentParser()
    #
    # parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
    #                     help='base directory path of program files')
    # parser.add_argument('--out', '-o', default='results-jamit/',
    #                     help='Directory to output the result')
    # parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
    #                     help='Root directory path of input image')
    #
    # args = parser.parse_args()
    #
    # file_name = 'test/denoising/MicroCT.mhd'
    #
    # W = 250
    # H = 1
    # D = 110
    #
    # #make output dir
    # result_dir = os.path.join(args.base, args.out)
    # if not os.path.exists(result_dir):
    #     os.makedirs(result_dir)
    #
    # #load image data
    # print("train1.mhd load")
    # sitktrain = sitk.ReadImage(os.path.join(args.root,file_name))
    # train = sitk.GetArrayFromImage(sitktrain).astype("float32")#たしかこれ正規化済
    # print("train1.mhd load done")
    #
    # print(train.shape)
    #
    # count=0
    # for i in range(0,640-W):
    #     for j in range(0,160-H):
    #         for k in range(0,480-D):
    #             # print("i=",i)
    #             # print("j=", j)
    #             # print("k=", k)
    #             # print("i+W=", i+W)
    #             # print("j+H=", j+H)
    #             # print("k+D=", k+D)
    #             check=train[k:k+D,j:j+H,i:i+W]
    #             std=np.std(check)
    #             # print(std)
    #             if std>34.0:
    #                 count+=1
    #                 print(count)



if __name__ == '__main__':
    main()



