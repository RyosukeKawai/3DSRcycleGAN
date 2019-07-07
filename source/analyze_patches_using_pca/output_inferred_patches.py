#coding:utf-8
import os, sys, time
import argparse, yaml, shutil, math
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pickle, glob
import chainer
from model import Generator

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '../..')))
import util.ioFunction_version_4_3 as IO
import util.yaml_utils  as yaml_utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                        help='base directory path of program files')
    parser.add_argument('--inputDir', '-i', default='',
                        help='path to input directory')
    parser.add_argument('--outputDir', '-o', default= '',
                        help='Directory to output the result')

    parser.add_argument('--model', '-m', default='',
                        help='Load model data(snapshot)')

    parser.add_argument('--min', default=255.,
                        help='Minimum value in LR image')
    parser.add_argument('--max', default=0,
                        help='Maxmum value in LR image')
    args = parser.parse_args()

    print('----- Save configs -----')
    result_dir = args.outputDir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    with open('{}/args_configs.yml'.format(result_dir), 'w') as f:
        f.write(yaml.dump(vars(args), default_flow_style=False))

    LR_MIN, LR_MAX = args.min, args.max

    print('----- Build model -----')
    print('GPU: {}'.format(args.gpu))
    print('')
    gen = Generator()
    chainer.serializers.load_npz(args.model, gen)
    if args.gpu >= 0:
        chainer.backends.cuda.set_max_workspace_size(1024 * 1024 * 1024) # 1GB
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
    xp = gen.xp

    fnames = glob.glob('{}/*.mhd'.format(args.inputDir))
    #print(fnames)
    print('----- Inference start -----')
    for i in fnames:
        fn = os.path.splitext(os.path.basename(i))[0]
        #Read data and reshape
        sitkTri = sitk.ReadImage(i)
        tri = sitk.GetArrayFromImage(sitkTri).astype("float32")
        tri = (tri-LR_MIN)/ (LR_MAX-LR_MIN)
        tri = chainer.Variable(xp.array(tri[np.newaxis, np.newaxis, :], dtype=xp.float32))
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            inferred_patch = gen(tri)

        # Generate probability map
        inferred_patch = inferred_patch.array
        if args.gpu >= 0:
            inferred_patch = chainer.backends.cuda.to_cpu(inferred_patch)
        inferred_patch = np.squeeze(inferred_patch, axis=(0,1))

        inferred_patch = ((inferred_patch+1)/2*255)
        sitkImg = sitk.GetImageFromArray(inferred_patch)
        sitkImg.SetSpacing([1,1,1])
        sitk.WriteImage(sitkImg, '{}/{}.mhd'.format(result_dir, fn))

    print('----- Inference done -----')

if __name__ == '__main__':
    main()
