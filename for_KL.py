#coding: utf-8
"""
* @auther ryosuke
* For completely unpairSR
"""

import pandas as pd
import util.yaml_utils  as yaml_utils
import os, sys
import argparse, yaml
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

def KLD(lr_hist, hr_hist, epsilon):

    lr_hist = (lr_hist + epsilon) / np.sum(lr_hist)
    hr_hist = (hr_hist + epsilon) / np.sum(hr_hist)

    return np.sum([li * np.log(li / hi) for li, hi in zip(lr_hist, hr_hist)])

def main():

        parser = argparse.ArgumentParser(description='Train pre')

        parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                            help='base directory path of program files')
        parser.add_argument('--config_path', type=str, default='configs/cutting_position.yml',
                            help='path to config file')
        parser.add_argument('--out', '-o', default= 'results/optimal_cutting_position_KL',
                            help='Directory to output the result')
        parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                            help='Root directory path of input image')

        args = parser.parse_args()

        config = yaml_utils.Config(yaml.load(open(os.path.join(args.base, args.config_path))))


        #make output dir
        result_dir = os.path.join(args.base, args.out)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        num_bin=0.4
        loop=115
        base=1570

        print('----- Start -----')
        for i in range(loop):
            i = i + base
            # load data
            lr = pd.read_csv(args.root + '\\LR_{}.csv'.format(i), names=('mean', 'std'))
            hr = pd.read_csv(args.root + '\\HR_{}.csv'.format(i), names=('mean', 'std'))

            # join
            x3 = np.hstack((lr['mean'], hr['mean']))
            y3 = np.hstack((lr['std'], hr['std']))

            # calc num of bins
            binx = int(float(abs(int(np.min(x3) - 1)) + int(np.max(x3) + 1)) * num_bin)
            biny = int(float(abs(int(np.min(y3) - 1)) + int(np.max(y3) + 1)) * num_bin)

            # make two histograms
            hist_lr, _, _ = np.histogram2d(lr['mean'], lr['std'], bins=[binx, biny],
                                           range=[[int(np.min(x3) - 1.), int(np.max(x3) + 1.)],
                                                  [int(np.min(y3) - 1.), int(np.max(y3) + 1.)]])
            hist_hr, _, _ = np.histogram2d(hr['mean'], hr['std'], bins=[binx, biny],
                                           range=[[int(np.min(x3) - 1.), int(np.max(x3) + 1.)],
                                                  [int(np.min(y3) - 1.), int(np.max(y3) + 1.)]])

            # calc epshiron
            lr_ep = 1. / (len(lr['mean']) * 10.)
            hr_ep = 1. / (len(hr['mean']) * 10.)

            # calc histogram base KL divergence
            kld = KLD(hist_lr, hist_hr, (lr_ep + hr_ep) / 2.)

            # save info
            df = pd.DataFrame({'ep': [(lr_ep + hr_ep) / 2.],'num': [i], 'KLD': [kld]})
            df.to_csv('{}/results_after_debug.csv'.format(result_dir), index=False, encoding='utf-8', mode='a',header=None)
        print('----- Finish -----')




        # print("mean",min_m,"@@",max_m)
        # print("std", min, "@@", max)

if __name__ == '__main__':
    main()


