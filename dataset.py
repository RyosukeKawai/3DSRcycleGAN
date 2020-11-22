#coding:utf-8
"""
* @auther tozawa
* @history
* 20180122
* "https://github.com/pfnet-research/chainer-pix2pix/blob/master/facade_dataset.py"
"""
import os, sys, time
import numpy as np
import pandas as pd
import chainer
import utils.ioFunctions as IO

class SrganDataset(chainer.dataset.DatasetMixin):
    def __init__(self, root, path, patch_side, min_max=[]):
        print(' Initilaze dataset ')

        self._root = root
        self._patch_side = patch_side
        self._patch_size = int(self._patch_side**3)
        self._min, self._max = min_max
        self._num_labels = 2

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
        coordinate_csv_path = path_pairs[0][0]
        self._coordinate = pd.read_csv(os.path.join(self._root, coordinate_csv_path), names=("x","y","z")).values.tolist()

        self._dataset=[]
        for i in path_pairs[1:]:
            print('   Tri from: {}'.format(i[0]))
            print('   Org from: {}'.format(i[1]))
            print('   Label from: {}'.format(i[2]))

            #Read data and reshape
            lr = (IO.read_mhd_and_raw(os.path.join(self._root, i[0])).astype(np.float32)-self._min)/(self._max-self._min) # =>[0, 1]
            hr = (IO.read_mhd_and_raw(os.path.join(self._root, i[1])).astype(np.float32)/127.5)-1. # x/255*2-1 => [-1, 1]
            label = IO.read_mhd_and_raw(os.path.join(self._root, i[2]))
            self._dataset.append((lr, hr, label))

        print(' Initilazation done ')

    def __len__(self):
        return len(self._coordinate)

    def get_example(self, i):
        """
        # return (lr, hr)
        # I assume length of dataset is one
        """
        x, y, z = self._coordinate[i]
        x_s, x_e = x, x + self._patch_side
        y_s, y_e = y, y + self._patch_side
        z_s, z_e = z, z + self._patch_side

        label_ = self._dataset[0][2][z_s:z_e, y_s:y_e, x_s:x_e].flatten()
        label = np.zeros((label_.size, self._num_labels), dtype=int)
        #"https://stackoverflow.com/questions/29831489/numpy-1-hot-array"
        label[np.arange(label_.size), label_] = 1
        label = label.transpose().reshape(self._num_labels, self._patch_side, self._patch_side, self._patch_side).astype(np.float32)

        return self._dataset[0][0][np.newaxis, z_s:z_e, y_s:y_e, x_s:x_e], self._dataset[0][1][np.newaxis, z_s:z_e, y_s:y_e, x_s:x_e], label

class InterpolateDataset(chainer.dataset.DatasetMixin):
    def __init__(self, root, path, patch_side, min_max=[]):
        print(' Initilaze dataset ')

        self._root = root
        self._patch_side = patch_side
        self._patch_size = int(self._patch_side**3)
        self._min, self._max = min_max
        self._upsampling_rate = 8

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
        coordinate_csv_path = path_pairs[0][0]
        self._coordinate = pd.read_csv(os.path.join(self._root, coordinate_csv_path), names=("x","y","z")).values.tolist()

        self._dataset=[]
        print('   LR from: {}'.format(path_pairs[1][0]))
        print('   HR from: {}'.format(path_pairs[1][1]))

        #Read data and reshape
        lr = ((IO.read_mhd_and_raw(os.path.join(self._root, path_pairs[1][0])).astype(np.float32)-self._min)/(self._max-self._min)) # =>[0, 1]
        hr = ((IO.read_mhd_and_raw(os.path.join(self._root, path_pairs[1][1])).astype(np.float32)-self._min)/(self._max-self._min))*2.-1. # x/255*2-1 => [-1, 1]
        self._dataset.append((lr, hr))

        print(' Initilazation done ')

    def __len__(self):
        return len(self._coordinate)

    def get_example(self, i):
        """
        # return (lr, hr)
        # I assume length of dataset is one
        """
        x, y, z = self._coordinate[i]
        x_s, x_e = x, x + self._patch_side
        y_s, y_e = y, y + self._patch_side
        z_s, z_e = z, z + self._patch_side
        hr_img = self._dataset[0][1][np.newaxis, z_s:z_e, y_s:y_e, x_s:x_e]

        lr_x = int(x/self._upsampling_rate+0.5)
        lr_y = int(y/self._upsampling_rate+0.5)
        lr_z = int(z/self._upsampling_rate+0.5)
        x_s, x_e = lr_x, lr_x + self._patch_side//self._upsampling_rate
        y_s, y_e = lr_y, lr_y + self._patch_side//self._upsampling_rate
        z_s, z_e = lr_z, lr_z + self._patch_side//self._upsampling_rate
        lr_img = self._dataset[0][0][np.newaxis, z_s:z_e, y_s:y_e, x_s:x_e]

        return lr_img, hr_img
