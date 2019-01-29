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
import argparse
import chainer
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '../..')))
import util.ioFunction_version_4_3 as IO

class CvaeDataset(chainer.dataset.DatasetMixin):
    def __init__(self, root, path, patch_side, min_max=[], augmentation=False):
        print(' Initilaze dataset ')

        self._root = root
        self._patch_side = patch_side
        self._patch_size = int(self._patch_side**3)
        self._min, self._max = min_max
        self._augmentation = augmentation

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
            print('   Org from: {}'.format(i[0]))

            #Read data and reshape
            hr = (IO.read_mhd_and_raw(os.path.join(self._root, i[0])).astype(np.float32)/127.5)-1. # x/255*2-1 => [-1, 1]
            self._dataset.append(hr)

        print(' Initilazation done ')

    def __len__(self):
        return len(self._coordinate)

    def transform(self, img1):
        # Random right left transform
        if np.random.rand() > 0.5:
            img1 = img1[:, :, :, ::-1]
        if np.random.rand() > 0.5:
            img1 = img1[:, :, ::-1, :]
        if np.random.rand() > 0.5:
            img1 = img1[:, ::-1, :, :]
        #img += np.random.uniform(size=img1.shape, low=0, high=1./128)
        return img1

    def get_example(self, i):
        """
        # return (lr, hr)
        # I assume length of dataset is one
        """
        x, y, z = self._coordinate[i]
        x_s, x_e = x, x + self._patch_side
        y_s, y_e = y, y + self._patch_side
        z_s, z_e = z, z + self._patch_side

        if self._augmentation:
            return self.transform(self._dataset[0][np.newaxis, z_s:z_e, y_s:y_e, x_s:x_e])

        return self._dataset[0][np.newaxis, z_s:z_e, y_s:y_e, x_s:x_e]
