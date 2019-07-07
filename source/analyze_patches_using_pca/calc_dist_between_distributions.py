#coding:utf-8
import os, sys, time
import argparse
import numpy as np
import pandas as pd
import math

def calc_kl_div(mu1, mu2, sigma1, sigma2):
    term1 = math.log(np.linalg.det(sigma2)/np.linalg.det(sigma1))
    term2 = np.trace(np.linalg.inv(sigma2).dot(sigma1))
    term3 = (mu1-mu2).transpose().dot(np.linalg.inv(sigma2)).dot(mu1-mu2)
    term4 = sigma1.shape[0]
    return 0.5*(term1+term2+term3-term4)

def calc_bhattacharyya_distance(mu1, mu2, sigma1, sigma2):
    """
    https://en.wikipedia.org/wiki/Bhattacharyya_distance
    """
    sigma = 0.5*(sigma1+sigma2)
    term1 = 0.125 * (mu1-mu2).transpose().dot(np.linalg.inv(sigma)).dot(mu1-mu2)
    term2 = 0.5 * math.log(np.linalg.det(sigma)/math.sqrt(np.linalg.det(sigma1)*np.linalg.det(sigma2)))

    return (term1 + term2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputCsvFile', '-i', help='input csv file')
    parser.add_argument('--outputDir', '-o', help='output directory')
    args = parser.parse_args()

    print('----- read csv file')
    df = pd.read_csv(args.inputCsvFile, names=('ax1','ax2','ax3'))
    gt_df = df[0:500]
    withSN_df = df[500:1000]
    withoutSN_df = df[1000:1500]

    print('----- gt vs withSN ------')
    sigma1 = gt_df.cov().values
    sigma2 = withSN_df.cov().values
    mu1 = gt_df.mean().values
    mu2 = withSN_df.mean().values
    #print(calc_kl_div(np.c_[mu1], np.c_[mu2], sigma1, sigma2))
    print(calc_bhattacharyya_distance(np.c_[mu1], np.c_[mu2], sigma1, sigma2))


    print('----- gt vs withoutSN ------')
    sigma1 = gt_df.cov().values
    sigma2 = withoutSN_df.cov().values
    mu1 = gt_df.mean().values
    mu2 = withoutSN_df.mean().values
    #print(calc_kl_div(np.c_[mu1], np.c_[mu2], sigma1, sigma2))
    print(calc_bhattacharyya_distance(np.c_[mu1], np.c_[mu2], sigma1, sigma2))


if __name__ == '__main__':
    main()
