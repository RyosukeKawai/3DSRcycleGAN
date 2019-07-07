# coding :utf-8
'''
* Apply singular value decomposition to convolutional layer weight.
* https://qiita.com/mizunototori/items/38291518110849e91a4c
* https://qiita.com/kyoro1/items/4df11e933e737703d549
* @auther tozawa
* @date 2018-7-18
'''
import os, sys, time
import argparse, six
import numpy as np
import matplotlib.pyplot as plt
import chainer
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '..')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', default='',
                        help='Load model data(snapshot)')
    parser.add_argument('--output_dir', '-o',
                        help='Output directory')
    args = parser.parse_args()

    param_name = ['conv1/W',
                    'conv2/W',
                    'conv3/W',
                    'conv4/W',
                    'conv5/W',
                    'conv6/W',
                    'conv7/W',
                    'conv8/W']

    param = np.load(args.model)
    for key, arr in six.iteritems(param):
        if key in param_name:
            print('Parameter name: {}'.format(key))
            weight = arr.reshape(arr.shape[0], -1)
            U, S, V = np.linalg.svd(weight)
            squared_sigma = S**2
            squared_sigma = squared_sigma / squared_sigma.max()
            plt.figure()
            plt.plot(range(0,len(squared_sigma)), squared_sigma, marker='o')
            plt.xlabel(r'Index of $\sigma$', fontsize=12)
            plt.ylabel(r'$\sigma^2$', fontsize=12)
            plt.title(key)
            plt.ylim((0.0, 1.1))
            plt.xlim((0.0, len(squared_sigma)-1))
            plt.xticks([0, int(len(squared_sigma)/2+.5), len(squared_sigma)-1])
            name,_ = key.split('/')
            plt.savefig(os.path.join(args.output_dir, 'fig_'+ name +'.png'))
            #plt.show()

if __name__ == '__main__':
    main()
