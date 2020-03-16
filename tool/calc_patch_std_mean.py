#coding: utf-8
"""
* @auther ryosuke
* reference source :https://github.com/zEttOn86/3D-SRGAN
"""

import os, sys, time, random
import matplotlib.pyplot as plt
import numpy as np
import argparse, yaml, shutil
import chainer
import pandas as pd
import utils.ioFunction_version_4_3 as IO
import SimpleITK as sitk

sys.path.append(os.path.dirname(__file__))


from dataset import CycleganDataset
import utils.yaml_utils  as yaml_utils


def main():
    parser = argparse.ArgumentParser(description='Train pre')

    parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                        help='base directory path of program files')
    parser.add_argument('--config_path', type=str, default='configs/Scatter_plot.yml',
                        help='path to config file')
    parser.add_argument('--out', '-o', default= 'results/Scatter_plot/old',
                        help='Directory to output the result')
    parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Root directory path of input image')

    args = parser.parse_args()

    config = yaml_utils.Config(yaml.load(open(os.path.join(args.base, args.config_path))))


    #make output dir
    result_dir = os.path.join(args.base, args.out)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)


    print('----- Load dataset -----')
    # Read path to hr data and lr data
    path_pairs = []
    with open(os.path.join(args.base, config.dataset['training_fn'])) as paths_file:
        for line in paths_file:
            line = line.split()
            if not line: continue
            path_pairs.append(line[:])

    coordinate_csv_path = path_pairs[0][0]  # LR
    coordinate_csv_path2 = path_pairs[0][1]  # HR
    coordinate = pd.read_csv(os.path.join(args.root, coordinate_csv_path), names=("x", "y", "z")).values.tolist()
    coordinate2 = pd.read_csv(os.path.join(args.root, coordinate_csv_path2),names=("x", "y", "z")).values.tolist()

    # data load
    sitklr = sitk.ReadImage(os.path.join(args.root,path_pairs[1][0]))
    lr = sitk.GetArrayFromImage(sitklr).astype("float32")
    sitkhr = sitk.ReadImage(os.path.join(args.root,path_pairs[1][1]))
    hr = sitk.GetArrayFromImage(sitkhr).astype("float32")

    print('----- data load done  -----')
    print('-----  start  -----')


    for i in range(len(coordinate)):
        x, y, z = coordinate[i]
        x_s, x_e = x, x + config.patch['patchside']
        y_s, y_e = y, y + config.patch['patchside']
        z_s, z_e = z, z + config.patch['patchside']

        #patch image
        patch_image = lr[z_s:z_e, y_s:y_e, x_s:x_e]

        #calc std and mean
        Std_lr = np.std(patch_image)
        mean_lr = np.mean(patch_image)

        #df = pd.DataFrame({'x': [x], 'y': [y], 'z': [z], 'std': [Std_lr], 'mean': [mean_lr]})
        df = pd.DataFrame({ 'mean': [mean_lr],'std': [Std_lr]})
        df.to_csv('{}/lr_std_mean.csv'.format(result_dir), index=False, header=False ,encoding='utf-8', mode='a')



    print('-----  LR done  -----')

    for j in range(len(coordinate2)):
        x, y, z = coordinate2[j]
        x_s, x_e = x, x + config.patch['patchside']
        y_s, y_e = y, y + config.patch['patchside']
        z_s, z_e = z, z + config.patch['patchside']

        #patch image
        patch_image2 = hr[z_s:z_e, y_s:y_e, x_s:x_e]

        #calc std and mean
        Std_hr=np.std(patch_image2)
        mean_hr=np.mean(patch_image2)
        # df = pd.DataFrame({'x':[x],'y':[y], 'z':[z], 'std':[Std_hr], 'mean':[mean_hr]})
        df = pd.DataFrame({ 'mean': [mean_hr],'std': [Std_hr]})
        df.to_csv('{}/hr_std_mean.csv'.format(result_dir), index=False, header=False,encoding='utf-8', mode='a')

    print('-----  HR done  -----')

if __name__ == '__main__':
    main()
