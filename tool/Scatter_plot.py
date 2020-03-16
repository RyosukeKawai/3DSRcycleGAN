#coding: utf-8
"""
* @auther ryosuke

"""

import os, sys, time, random
import matplotlib.pyplot as plt
import numpy as np
import argparse, yaml, shutil
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy.stats import gaussian_kde,multivariate_normal



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





    #-------------------data----------------------#
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
    # lr_patch = pd.read_csv(os.path.join(args.root, lr_path_csv),names=('mean','std'))
    # hr_patch = pd.read_csv(os.path.join(args.root, hr_patch_csv),names=('mean','std'))

    # new
    #lr_patch = pd.read_csv("G:/experiment/input_adjast_result/for_KL/LR_1680.csv",names=('mean', 'std'))
    #hr_patch = pd.read_csv("G:/experiment/input_adjast_result/for_KL/HR_1680.csv",names=('mean', 'std'))

    #undergraduate
    #lr_patch = pd.read_csv("G:/data/train/denoising/undergraduate/train1/lr_std_mean.csv",names=('mean', 'std'))
    #hr_patch = pd.read_csv("G:/data/train/denoising/undergraduate/train2/hr_std_mean.csv",names=('mean', 'std'))

    #undergraduate
    #lr_patch = np.loadtxt("G:/data/train/denoising/undergraduate/train1/lr_std_mean.csv", dtype=float, delimiter=",")
    #hr_patch = np.loadtxt("G:/data/train/denoising/undergraduate/train2/hr_std_mean.csv", dtype=float, delimiter=",")

    # new
    #lr_patch = np.loadtxt("G:/experiment/input_adjast_result/for_KL/LR_1680.csv",dtype=float,delimiter=",")
    #hr_patch = np.loadtxt("G:/experiment/input_adjast_result/for_KL/HR_1680.csv", dtype=float, delimiter=",")
    #----------------------------------------------------------------------------------------------------------#


    #-------------------------make 2d histogram-----------------#

    binx = 36
    biny = 80
    fig = plt.figure()

    #for heatmap
    #ax = fig.add_subplot(111)
    #H= ax.hist2d(lr_patch[:, 1], lr_patch[:, 0], bins=[binx, biny], range=([0, 90], [0, 200]),cmap='gist_ncar',norm=colors.LogNorm())
    #H= ax.hist2d(hr_patch[:, 1], hr_patch[:, 0], bins=[binx, biny], range=([0, 90], [0, 200]),cmap='gist_ncar',norm=colors.LogNorm(1,3000))

    # for bar3d
    #ax = fig.add_subplot(111,projection="3d")
    #H,_,_ = np.histogram2d(lr_patch[:,0],lr_patch[:,1],bins=[binx,biny],range=([0,200],[0,90]))
    #H, _, _ = np.histogram2d(hr_patch[:, 0], hr_patch[:, 1], bins=[binx, biny], range=([0, 200], [0, 90]))

    #--------------------------------------------------------------------------------------------#

    #----------3d scatterPlot---------------#
    # X1,X2 = np.meshgrid(lr_patch[:,1],lr_patch[:,0])
    # X_plot = np.c_[X1.flatten(),X2.flatten()]
    # y=multivariate_normal.pdf(X_plot)
    # ax.scatter3D(lr_patch[:, 1],lr_patch[:, 0],)


    #---------heatmap------------------------#
    # """https://codeday.me/jp/qa/20190112/133965.html"""
    # lh=np.vstack([lr_patch[:,1],lr_patch[:,0]])
    # z =gaussian_kde(lh)(lh)
    # lh=np.vstack([hr_patch[:,1],hr_patch[:,0]])
    # z =gaussian_kde(lh)(lh)
    # idx =z.argsort()
    # lr_patch[:, 0],lr_patch[:,1],z=lr_patch[:, 0][idx],lr_patch[:,1][idx],z[idx]
    # hr_patch[:, 0],hr_patch[:,1],z=hr_patch[:, 0][idx],hr_patch[:,1][idx],z[idx]
    #---------------------------------------------------------------------------#

    #------------for Three-dimensional display of two-dimensional histogram colorbar------------#
    # xdata, ydata = np.meshgrid(np.arange(H.shape[1]),np.arange(H.shape[0]))
    # xdata=xdata.flatten()
    # ydata=ydata.flatten()
    # zdata=H.flatten()
    #
    # #color
    # norm = colors.Normalize(0, 3000)
    # color_values = cm.gist_ncar(norm(zdata))


    #Three-dimensional display of two-dimensional histogram colorbar
    # v=np.linspace(0,400,10,endpoint=True)
    # colormap=plt.cm.ScalarMappable(cmap=cm.gist_ncar)
    # colormap.set_array(np.zeros(len(zdata)))
    # colorbar = plt.colorbar(colormap).set_label('frequency')

    #-------------------------------------------------------------------------------------------#

    #--------------plot-----------------#

    #for scatter plot
    #plt.xlim(0,90)
    #plt.ylim(0, 200)
    #plt.scatter(lr_patch['std'],lr_patch['mean'], c='blue' )
    #plt.scatter(hr_patch['std'],hr_patch['mean'], c='red' )

    #for heatmap
    # # plt.scatter(lr_patch[:,1],lr_patch[:,0],c=z)
    # heatmap=plt.scatter(hr_patch[:,1],hr_patch[:,0],c=z)
    # # v=np.linspace(0,1800,10,endpoint=True)
    # fig.colorbar(H[3],ax=ax)

    #for bar3d
    #ax.bar3d(xdata, ydata, np.zeros(len(zdata)), 1, 1, zdata,color=color_values,zsort='max')


    #for 2d histogram
    #color
    # H[3].set_clim(0,3000)
    # norm = colors.LogNorm(0, 3000)
    # color_values = cm.gist_ncar(norm(H[0]))
    #print(np.max(H[0]))
    #v =(np.linspace(1, 3000, 30, endpoint=True))
    # fig.colorbar(H[3],cmap=color_values, ax=ax).set_label('frequency')
    #fig.colorbar(H[3],ax=ax).set_label('frequency')


    #plotconfig
    # plt.xlim(0,90)
    # plt.ylim(0,200)
    # ax.set_xticks(np.linspace(0,90,10))
    # ax.set_yticks(np.linspace(0,200,5))
    # plt.pcolor(vmin=0,vmax=300,)

    #old
    plt.title('undergraduate train_LR TOTAL 41529 patch')
    #plt.title('undergraduate train_HR TOTAL 48994 patch')

    #new
    #plt.title('z=1680 train_LR TOTAL 45724 patch')
    #plt.title('z=1680 train_HR TOTAL 44355 patch')

    plt.ylabel('mean')
    plt.xlabel('std')
    #ax.set_zlabel("frequency")
    plt.show()

if __name__ == '__main__':
    main()


