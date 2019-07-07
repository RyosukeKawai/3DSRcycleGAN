#coding: utf-8
"""
* @auther ryosuke
* For completely unpairSR
"""

import pandas as pd
import util.yaml_utils  as yaml_utils
import os, sys, math
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
        parser.add_argument('--out', '-o', default= 'results/histogram_base_KL_debug',
                            help='Directory to output the result')
        parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                            help='Root directory path of input image')

        args = parser.parse_args()

        config = yaml_utils.Config(yaml.load(open(os.path.join(args.base, args.config_path))))


        #make output dir
        result_dir = os.path.join(args.base, args.out)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        mu1=[-10,0]
        mu2=[10,0]
        cov=[[2,0],[0,2]]
        total = 45000#num of data
        loop =100# num of loop
        epshiron = [2.22e-6,2.22e-7,2.22e-8]
        num_bins = np.arange(0.1,5.0,0.1,dtype=float)

        #Scatter_plot
        t1, v1 = np.random.multivariate_normal(mu1, cov, total).T
        t2, v2 = np.random.multivariate_normal(mu2, cov, total).T
        plt.scatter(t1, v1, c='cyan')
        plt.scatter(t2, v2, c='green')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.show()

        print('----- Start -----')
        for e in range(len(epshiron)):
                ep=epshiron[e]
                kld=np.zeros(loop)
                x_bin=np.zeros(loop)
                y_bin=np.zeros(loop)
                for num_b in range(len(num_bins)):
                        num_bin=num_bins[num_b]
                        for i in range(loop):

                                #make datas
                                x1, y1 = np.random.multivariate_normal(mu1, cov, total).T
                                x2, y2 = np.random.multivariate_normal(mu2, cov, total).T

                                # join
                                x3 = np.hstack((x1, x2))
                                y3 = np.hstack((y1, y2))

                                #calc num of bins
                                binx = int(float(abs(int(np.min(x3) - 1)) + int(np.max(x3) + 1)) * num_bin)
                                biny = int(float(abs(int(np.min(y3) - 1)) + int(np.max(y3) + 1)) * num_bin)
                                x_bin[i]=binx
                                y_bin[i]=biny

                                # make two histograms
                                hist_lr, _, _ = np.histogram2d(x1, y1, bins=[binx, biny], range=[[int(np.min(x3)-1.),int(np.max(x3)+1.)], [int(np.min(y3)-1.),int(np.max(y3)+1.)]])
                                hist_hr, _, _ = np.histogram2d(x2, y2, bins=[binx, biny], range=[[int(np.min(x3)-1.),int(np.max(x3)+1.)], [int(np.min(y3)-1.),int(np.max(y3)+1.)]])
                                if (np.sum(hist_hr) != total or np.sum(hist_lr) != total): sys.exit()

                                #calc histogram base KL divergence
                                kld[i]=KLD(hist_lr, hist_hr,ep)

                        #save info
                        df = pd.DataFrame({'ep': [ep], 'num_b': [num_bin],'KLD_mean': [np.mean(kld)],'KLD_std': [np.std(kld)],
                                           'binx_mean': [np.mean(x_bin)], 'binx_std': [np.std(x_bin)],'biny_mean': [np.mean(y_bin)], 'biny_std': [np.std(y_bin)]})
                        if(e==0 and num_b==0.1 ):df.to_csv('{}/results.csv'.format(result_dir), index=False, encoding='utf-8', mode='a',header=True)
                        df.to_csv('{}/results_another3.csv'.format(result_dir), index=False, encoding='utf-8', mode='a',header=None)

        print('----- Finish -----')

if __name__ == '__main__':
    main()


# main
# print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
# print(np.min(x3))
# print(np.max(x3))
# print(np.min(y3))
# print(np.max(y3))
#
#
# print(int(np.min(x3)-1))
# print(int(np.max(x3) + 1.))
# print(int(np.min(y3) - 1.))
# print(int(np.max(y3) + 1.))
# print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

# print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
# print(hist_hr)
# print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
# print(hist_lr)
# print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")


#KLD
    # lr_hist = (lr_hist) / np.sum(lr_hist)
    # hr_hist = (hr_hist) / np.sum(hr_hist)

    # print("*************************************************")
    # print(hr_hist)
    # print("*************************************************")
    # print(lr_hist)
    # print("*************************************************")


    # k=np.zeros(len(lr_hist))
    # sm=np.zeros(len(lr_hist))
    # print(lr_hist.shape)
    # print(k.shape)
    # for li,hi in zip(lr_hist, hr_hist):
    #
    #     if(li.any()==0):
    #
    #     elif(hi.any()==0):
    #         k=li * np.log(li / epsilon)
    #     else:
    #         k=li * np.log(li / hi)
    #
    #     sm=sm+k

    # return sm