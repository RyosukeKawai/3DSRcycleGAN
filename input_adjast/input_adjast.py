#coding: utf-8

"""
* @auther ryosuke
* reference source :https://github.com/zEttOn86/3D-SRGAN

input :image(.mhd) , mask(.mhd) , Cutting position(.yml)
output: Cutting position and KL divergence(.csv)

"""

import pandas as pd
import util.yaml_utils  as yaml_utils
import os, sys
import argparse, yaml
import SimpleITK as sitk
import numpy as np




def main():
    parser = argparse.ArgumentParser(description='Train pre')

    parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                        help='base directory path of program files')
    parser.add_argument('--config_path', type=str, default='configs/cutting_position.yml',
                        help='path to config file')
    parser.add_argument('--out', '-o', default= 'results/cutting_position',
                        help='Directory to output the result')
    parser.add_argument('--margin', '-m', default=5,
                        help='patch margin (ratio)')
    parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Root directory path of input image')

    args = parser.parse_args()

    config = yaml_utils.Config(yaml.load(open(os.path.join(args.base, args.config_path))))

    #image and mask path
    image_path='train/denoising/MicroCT.mhd'
    mask_path='train/calc_mask.mhd'

    #make output dir
    result_dir = os.path.join(args.base, args.out)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    print('----- Load data -----')

    sitkhr = sitk.ReadImage(os.path.join(args.root,image_path))
    hr = sitk.GetArrayFromImage(sitkhr).astype("float32")
    sitkmask = sitk.ReadImage(os.path.join(args.root,mask_path))
    calc_mask = sitk.GetArrayFromImage(sitkmask).astype("float32")

    #adjast
    train = hr[:, 0:1210, :]
    mask = calc_mask[:, 0:1210, :]

    print('----- Loop start -----')

    hr_patch = []
    lr_patch = []
    for i in range(config.num['number']):
        i=i+config.num['start_position']

        # cut train and mask
        cut_hr , cut_lr  = np.split(train,[i], 0)
        hr_mask , lr_mask  = np.split(mask, [i], 0)

        # extract patch image

        # patch margin check

        #calc mean and std

        #calc KL divergence

        #save info csv




if __name__ == '__main__':
    main()