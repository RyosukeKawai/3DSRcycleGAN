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

    #li_sum = np.sum([li * np.log(li / hi) for li, hi in zip(lr_hist, hr_hist)])
    hi_sum = np.sum([hi * np.log(hi / li) for li, hi in zip(lr_hist, hr_hist)])
    return hi_sum

def main():

        parser = argparse.ArgumentParser(description='Train pre')

        parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                            help='base directory path of program files')
        parser.add_argument('--config_path', type=str, default='configs/cutting_position.yml',
                            help='path to config file')
        parser.add_argument('--out', '-o', default= 'results/covariance',
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
            o = i + base
            # load data
            lr= np.loadtxt(args.root + '\\LR_{}.csv'.format(o), dtype=float,delimiter=",")
            hr= np.loadtxt(args.root + '\\HR_{}.csv'.format(o), dtype=float, delimiter=",")

            #lr= np.loadtxt("G:/data/train/denoising/undergraduate/train1/lr_std_mean.csv", dtype=float,delimiter=",")
            #hr= np.loadtxt("G:/data/train/denoising/undergraduate/train2/hr_std_mean.csv", dtype=float, delimiter=",")

            #calc
            # covariance
            # Sxy=1/n * sigma(i=1 , n){(x-x.mean)(y-y.mean())}
            lr_mean_mean=lr[:,0].mean()
            lr_std_mean=lr[:,1].mean()
            total = 0.
            hr_mean_mean=hr[:,0].mean()
            hr_std_mean=hr[:,1].mean()
            total2 = 0.



            #covariance
            for j in range(len(lr[:,0])):
                total = total + (lr[j,0]-lr_mean_mean)*(lr[j,1]-lr_std_mean)
            lr_S = total* 1./float(len(lr[:,0]))

            for k in range(len(hr[:, 0])):
                total2 = total2 + (hr[k, 0] - hr_mean_mean) * (hr[k, 1] - hr_std_mean)
            hr_S= total2 * 1. / float(len(hr[:, 0]))

            #Correlation coefficient
            lr_C = lr_S/(lr[:,0].std()*lr[:,1].std())
            hr_C = hr_S /(hr[:, 0].std() * hr[:, 1].std())

            #Marginal distribution
            lr_mean_std=lr[:,0].std()
            lr_std_std=lr[:,1].std()
            hr_mean_std=hr[:,0].std()
            hr_std_std=hr[:,1].std()

            # # save info
            df = pd.DataFrame({'z': [o],'lr_S': [lr_S], 'hr_S': [hr_S],'difference_s': [abs(lr_S-hr_S)],'lr_c':[lr_C],'hr_c': [hr_C],'difference_c': [abs(lr_C-hr_C)],
                               'lr_mean_mean':[lr_mean_mean],'lr_mean_std':[lr_mean_std],'lr_std_mean':[lr_std_mean],'lr_std_std':[lr_std_std],'hr_mean_mean':[hr_mean_mean],
                               'hr_mean_std':[hr_mean_std],'hr_std_mean':[hr_std_mean],'hr_std_std':[hr_std_std]})
            if o==base:df.to_csv('{}/covariance.csv'.format(result_dir), index=False, encoding='utf-8', mode='a',header=True)
            else : df.to_csv('{}/covariance.csv'.format(result_dir), index=False, encoding='utf-8', mode='a', header=None)
            #df.to_csv('{}/covariance_undergraduate.csv'.format(result_dir), index=False, encoding='utf-8', mode='a', header=True)


        print('----- Finish -----')




        # print("mean",min_m,"@@",max_m)
        # print("std", min, "@@", max)

if __name__ == '__main__':
    main()


