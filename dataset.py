#coding:utf-8
"""
* @auther ryosuke
* reference source :https://github.com/zEttOn86/3D-SRGAN
* completely unpairSR
"""
import os, sys, time,random,csv
import numpy as np
import pandas as pd
import chainer
import util.ioFunction_version_4_3 as IO
import argparse
import random

class CycleganDataset(chainer.dataset.DatasetMixin):
    def __init__(self, root, path, patch_side, min_max=[]):
        print(' Initilaze dataset ')

        self._root = root
        self._patch_side = patch_side
        self._patch_size = int(self._patch_side**3)
        self._min, self._max = min_max

        # self._edge = 16

        # Read path to hr data and lr data
        path_pairs = []
        with open(path) as paths_file:
            for line in paths_file:
                line = line.split()
                if not line : continue
                path_pairs.append(line[:])

        """
        * Read coordinate csv
        fugafuga.csv
        x1,y1,z1
        x2,y2,z2
        ...
        xn,yn,zn
        """
        coordinate_csv_path = path_pairs[0][0]#LR 45724
        coordinate_csv_path2 = path_pairs[0][1]#HR 44355
        self._coordinate = pd.read_csv(os.path.join(self._root, coordinate_csv_path), names=("x","y","z")).values.tolist()
        self._coordinate2 = pd.read_csv(os.path.join(self._root, coordinate_csv_path2),
                                       names=("x", "y", "z")).values.tolist()
        self._coordinate_add = random.sample(self._coordinate,len(self._coordinate)-len(self._coordinate2))
        self._coordinate2 = self._coordinate2+self._coordinate_add





        self._dataset=[]
        for i in path_pairs[1:]:
            print('   Tri from: {}'.format(i[0]))
            print('   Org from: {}'.format(i[1]))

            #Read data and reshape
            lr = (IO.read_mhd_and_raw(os.path.join(self._root, i[0])).astype(np.float32)/127.5)-1.
            hr = (IO.read_mhd_and_raw(os.path.join(self._root, i[1])).astype(np.float32)/127.5)-1. # x/255*2-1 => [-1, 1]
            # margin = ((8, self._patch_side),
            #          (8, self._patch_side),
            #          (8, self._patch_side))
            # lr = np.pad(lr, margin, 'edge')
            # hr = np.pad(hr, margin, 'edge')
            self._dataset.append((lr,hr))


        print(' Initilazation done ')

    def __len__(self):
        return len(self._coordinate)

    def transform(self, img1, img2):
        # Random right left transform
        if np.random.rand() > 0.5:
            img1 = img1[:, :, :, ::-1]
            img2 = img2[:, :, :, ::-1]
        if np.random.rand() > 0.5:
            img1 = img1[:, :, ::-1, :]
            img2 = img2[:, :, ::-1, :]
        if np.random.rand() > 0.5:
            img1 = img1[:, ::-1, :, :]
            img2 = img2[:, ::-1, :, :]
        #img += np.random.uniform(size=img1.shape, low=0, high=1./128)
        return img1, img2

    def get_example(self, i):
        """
        # return (lr, hr)
        # I assume length of dataset is one
        """
        #lr
        x, y, z = self._coordinate[i]
        x_s, x_e = x, x + 8
        y_s, y_e = y, y + 8
        z_s, z_e = z, z + 8

        # hr
        x_h, y_h, z_h = self._coordinate2[i]
        x_sh, x_eh = x_h, x_h + self._patch_side
        y_sh, y_eh = y_h, y_h + self._patch_side
        z_sh, z_eh = z_h, z_h + self._patch_side

        return self._dataset[0][0][np.newaxis, z_s:z_e, y_s:y_e, x_s:x_e], \
               self._dataset[0][1][np.newaxis, z_sh:z_eh, y_sh:y_eh, x_sh:x_eh]