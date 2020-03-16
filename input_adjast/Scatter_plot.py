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
import util.ioFunction_version_4_3 as IO
import SimpleITK as sitk

sys.path.append(os.path.dirname(__file__))


from dataset import CycleganDataset
import util.yaml_utils  as yaml_utils


def main():
    parser = argparse.ArgumentParser(description='Train pre')

    parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                        help='base directory path of program files')
    parser.add_argument('--config_path', type=str, default='configs/Scatter_plot.yml',
                        help='path to config file')
    parser.add_argument('--out', '-o', default= 'results/Scatter_plot',
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
    # path_pairs = []
    # with open(os.path.join(args.base, config.dataset['training_fn'])) as paths_file:
    #     for line in paths_file:
    #         line = line.split()
    #         if not line: continue
    #         path_pairs.append(line[:])
    #
    # lr_path_csv = path_pairs[2][0]  # LR
    # hr_patch_csv = path_pairs[2][1]  # HR
    # lr_patch = pd.read_csv(os.path.join(args.root, lr_path_csv))
    # hr_patch = pd.read_csv(os.path.join(args.root, hr_patch_csv))
    lr_patch = pd.read_csv("G:/experiment/input_adjast_result/for_KL/LR_1680.csv")
    hr_patch = pd.read_csv("G:/experiment/input_adjast_result/for_KL/HR_1680.csv")

    lr_patch.describe()
    hr_patch.describe()

    #plot
    plt.scatter(lr_patch['std'],lr_patch['mean'],c='blue')
    plt.scatter(hr_patch['std'],hr_patch['mean'], c='red' )
    # plt.scatter(lr_patch['std'],lr_patch['mean'],c='blue')

    plt.ylabel('mean')
    plt.xlabel('std')
    plt.show()








if __name__ == '__main__':
    main()
