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
    parser.add_argument('--out', '-o', default='results2/',
                        help='Directory to output the result')
    parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Root directory path of input image')

    args = parser.parse_args()

    file_name='train/denoising/train2/train2.mhd'

    #make output dir
    result_dir = os.path.join(args.base, args.out)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    #load image data
    print("test.mhd load")
    sitktest = sitk.ReadImage(os.path.join(args.root,file_name))
    test = sitk.GetArrayFromImage(sitktest).astype("float32")#たしかこれ正規化済
    print("test.mhd load done")

    print(test.shape)

    # train2 cut
    testpatch=test[400:510,140:150,496:746]
    testpatch=testpatch.flatten()
    IO.write_mhd_and_raw_withoutSitk(testpatch, result_dir + '/for_jamit/Consideration/train2/con1.mhd',
                                         ndims=3, size=[250, 10, 110],
                                         space=[0.07, 0.066, 0.07])

    testpatch2=test[400:510,150:160,496:746]
    testpatch2=testpatch2.flatten()
    IO.write_mhd_and_raw_withoutSitk(testpatch2, result_dir + '/for_jamit/Consideration/train2/con2.mhd',
                                         ndims=3, size=[250, 10, 110],
                                         space=[0.07, 0.066, 0.07])

    testpatch3=test[500:610,120:130,496:746]
    testpatch3=testpatch3.flatten()
    IO.write_mhd_and_raw_withoutSitk(testpatch3, result_dir + '/for_jamit/Consideration/train2/con3.mhd',
                                         ndims=3, size=[250, 10, 110],
                                         space=[0.07, 0.066, 0.07])

    testpatch4=test[600:710,150:160,744:994]
    testpatch4=testpatch4.flatten()
    IO.write_mhd_and_raw_withoutSitk(testpatch4, result_dir + '/for_jamit/Consideration/train2/con4.mhd',
                                         ndims=3, size=[250, 10, 110],
                                         space=[0.07, 0.066, 0.07])

    testpatch5=test[600:710,160:170,744:994]
    testpatch5=testpatch5.flatten()
    IO.write_mhd_and_raw_withoutSitk(testpatch5, result_dir + '/for_jamit/Consideration/train2/con5.mhd',
                                         ndims=3, size=[250, 10, 110],
                                         space=[0.07, 0.066, 0.07])

    testpatch=test[600:710,170:180,744:994]
    testpatch=testpatch.flatten()
    IO.write_mhd_and_raw_withoutSitk(testpatch, result_dir + '/for_jamit/Consideration/train2/con6.mhd',
                                         ndims=3, size=[250, 10, 110],
                                         space=[0.07, 0.066, 0.07])

    testpatch=test[700:810,130:140,744:994]
    testpatch=testpatch.flatten()
    IO.write_mhd_and_raw_withoutSitk(testpatch, result_dir + '/for_jamit/Consideration/train2/con7.mhd',
                                         ndims=3, size=[250, 10, 110],
                                         space=[0.07, 0.066, 0.07])

    testpatch=test[700:810,140:150,744:994]
    testpatch=testpatch.flatten()
    IO.write_mhd_and_raw_withoutSitk(testpatch, result_dir + '/for_jamit/Consideration/train2/con8.mhd',
                                         ndims=3, size=[250, 10, 110],
                                         space=[0.07, 0.066, 0.07])

    testpatch=test[700:810,150:160,744:994]
    testpatch=testpatch.flatten()
    IO.write_mhd_and_raw_withoutSitk(testpatch, result_dir + '/for_jamit/Consideration/train2/con9.mhd',
                                         ndims=3, size=[250, 10, 110],
                                         space=[0.07, 0.066, 0.07])

    testpatch=test[800:910,10:20,496:746]
    testpatch=testpatch.flatten()
    IO.write_mhd_and_raw_withoutSitk(testpatch, result_dir + '/for_jamit/Consideration/train2/con10.mhd',
                                         ndims=3, size=[250, 10, 110],
                                         space=[0.07, 0.066, 0.07])

    testpatch=test[800:910,20:30,496:746]
    testpatch=testpatch.flatten()
    IO.write_mhd_and_raw_withoutSitk(testpatch, result_dir + '/for_jamit/Consideration/train2/con11.mhd',
                                         ndims=3, size=[250, 10, 110],
                                         space=[0.07, 0.066, 0.07])

    testpatch=test[900:1010,50:60,496:746]
    testpatch=testpatch.flatten()
    IO.write_mhd_and_raw_withoutSitk(testpatch, result_dir + '/for_jamit/Consideration/train2/con12.mhd',
                                         ndims=3, size=[250, 10, 110],
                                         space=[0.07, 0.066, 0.07])
    testpatch=test[1000:1110,0:10,744:994]
    testpatch=testpatch.flatten()
    IO.write_mhd_and_raw_withoutSitk(testpatch, result_dir + '/for_jamit/Consideration/train2/con13.mhd',
                                         ndims=3, size=[250, 10, 110],
                                         space=[0.07, 0.066, 0.07])
    testpatch=test[1000:1110,10:20,744:994]
    testpatch=testpatch.flatten()
    IO.write_mhd_and_raw_withoutSitk(testpatch, result_dir + '/for_jamit/Consideration/train2/con14.mhd',
                                         ndims=3, size=[250, 10, 110],
                                         space=[0.07, 0.066, 0.07])
    testpatch=test[2600:2710,0:10,744:994]
    testpatch=testpatch.flatten()
    IO.write_mhd_and_raw_withoutSitk(testpatch, result_dir + '/for_jamit/Consideration/train2/con15.mhd',
                                         ndims=3, size=[250, 10, 110],
                                         space=[0.07, 0.066, 0.07])
    testpatch=test[2600:2710,10:20,744:994]
    testpatch=testpatch.flatten()
    IO.write_mhd_and_raw_withoutSitk(testpatch, result_dir + '/for_jamit/Consideration/train2/con16.mhd',
                                         ndims=3, size=[250, 10, 110],
                                         space=[0.07, 0.066, 0.07])
    testpatch=test[2700:2810,80:90,496:746]
    testpatch=testpatch.flatten()
    IO.write_mhd_and_raw_withoutSitk(testpatch, result_dir + '/for_jamit/Consideration/train2/con17.mhd',
                                         ndims=3, size=[250, 10, 110],
                                         space=[0.07, 0.066, 0.07])
    testpatch=test[2800:2910,280:290,496:746]
    testpatch=testpatch.flatten()
    IO.write_mhd_and_raw_withoutSitk(testpatch, result_dir + '/for_jamit/Consideration/train2/con18.mhd',
                                         ndims=3, size=[250, 10, 110],
                                         space=[0.07, 0.066, 0.07])

    # train cut
    # for i in range(1,9):
    #     j=i
    #     i=125+25*(i-1)
    #     print("i=",i)
    #     test_patch=test[230:340,66:76,i:i+250]
    #     std=np.std(test_patch)
    #     mean=np.mean(test_patch)
    #     df=pd.DataFrame({'num':[j],'std':[std],'mean':[mean]})
    #     df.to_csv('{}/include_mean_results.csv'.format(result_dir),index=False, encoding='utf-8', mode='a')
    #     # print("num={} save".format(j))
    #     # test_patch=test_patch.flatten()
    #     # IO.write_mhd_and_raw_withoutSitk(test_patch, result_dir + '/for_jamit/test{}.mhd'.format(j),
    #     #                                  ndims=3, size=[250, 1, 110],
    #     #                                  space=[0.07, 0.066, 0.07])


if __name__ == '__main__':
    main()



