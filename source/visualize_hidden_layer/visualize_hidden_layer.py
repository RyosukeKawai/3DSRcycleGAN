#coding:utf-8
import os, sys, time
import argparse, yaml, shutil, math
import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pickle
import chainer

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '../..')))
import util.ioFunction_version_4_3 as IO
import util.yaml_utils  as yaml_utils
from model import Generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                        help='base directory path of program files')
    parser.add_argument('--config_path', type=str, default='../../configs/training.yml',
                        help='path to config file')
    parser.add_argument('--out', '-o', default= '../../work/visualize_hidden_layer',
                        help='Directory to output the result')

    parser.add_argument('--model', '-m', default='',
                        help='Load model data(snapshot)')

    parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Root directory path of input image')

    parser.add_argument('--save_flag', type=bool, default=False,
                        help='Decision flag whether to save criteria image')
    parser.add_argument('--filename_arg', type=str, default='val_fn',
                        help='Which do you want to use val_fn or test_fn')

    args = parser.parse_args()

    config = yaml_utils.Config(yaml.load(open(os.path.join(args.base, args.config_path))))
    print('GPU: {}'.format(args.gpu))
    LR_PATCH_SIDE, HR_PATCH_SIDE = config.patch['patchside'], config.patch['patchside']
    LR_PATCH_SIZE, HR_PATCH_SIZE = int(LR_PATCH_SIDE**3), int(HR_PATCH_SIDE**3)
    LR_MIN, LR_MAX = config.patch['lrmin'], config.patch['lrmax']
    print('HR PATCH SIZE: {}'.format(HR_PATCH_SIZE))
    print('LR PATCH SIZE: {}'.format(LR_PATCH_SIZE))
    print('')

    gen = Generator()
    chainer.serializers.load_npz(args.model, gen)
    if args.gpu >= 0:
        chainer.backends.cuda.set_max_workspace_size(1024 * 1024 * 1024) # 1GB
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
    xp = gen.xp

    def create_result_dir(base_dir, output_dir, config_path, config):
        """https://github.com/pfnet-research/sngan_projection/blob/master/train.py"""
        result_dir = os.path.join(base_dir, output_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        def copy_to_result_dir(fn, result_dir):
            bfn = os.path.basename(fn)
            shutil.copy(fn, '{}/{}'.format(result_dir, bfn))

        copy_to_result_dir(
            os.path.join(base_dir, config_path), result_dir)

        copy_to_result_dir(
            os.path.join(base_dir, '../..', config.network['fn']), result_dir)
        copy_to_result_dir(
            os.path.join(base_dir, '../..', config.updater['fn']), result_dir)
        copy_to_result_dir(
            os.path.join(base_dir, '../..', config.dataset[args.filename_arg]), result_dir)

    create_result_dir(args.base,  args.out, args.config_path, config)

    # Read data
    path_pairs = []
    with open(os.path.join(args.base, '../..', config.dataset[args.filename_arg])) as paths_file:
        for line in paths_file:
            line = line.split()
            if not line : continue
            path_pairs.append(line[:])

    print(' Inference start')
    for i in path_pairs:
        print('    Tri from: {}'.format(i[0]))
        print('    Org from: {}'.format(i[1]))
        #Read data and reshape
        sitkTri = sitk.ReadImage(os.path.join(args.root, i[0]))
        tri = sitk.GetArrayFromImage(sitkTri).astype("float32")
        tri = (tri-LR_MIN)/ (LR_MAX-LR_MIN)
        # Calculate maximum of number of patch at each side
        ze, ye, xe = tri.shape
        xm = int(math.ceil((float(xe)/float(config.patch['interval']))))
        ym = int(math.ceil((float(ye)/float(config.patch['interval']))))
        zm = int(math.ceil((float(ze)/float(config.patch['interval']))))

        margin = ((0, config.patch['patchside']),
                  (0, config.patch['patchside']),
                  (0, config.patch['patchside']))
        tri = np.pad(tri, margin, 'edge')
        tri = chainer.Variable(xp.array(tri[np.newaxis, np.newaxis, :], dtype=xp.float32))

        ch = 4
        sc_map = np.zeros((ch, ze+config.patch['patchside'],ye+config.patch['patchside'], xe+config.patch['patchside']))
        res_map = np.zeros((ch, ze+config.patch['patchside'],ye+config.patch['patchside'], xe+config.patch['patchside']))
        overlap_count = np.zeros((sc_map.shape[1], sc_map.shape[2], sc_map.shape[3]))

        # Patch loop
        print('     #Patches {}'.format(xm*ym*zm))
        for s in range(xm*ym*zm):
            xi = int(s%xm)*config.patch['interval']
            yi = int((s%(ym*xm))/xm)*config.patch['interval']
            zi = int(s/(ym*xm))*config.patch['interval']
            # Extract patch from original image
            patch = tri[:,:,zi:zi+config.patch['patchside'],yi:yi+config.patch['patchside'],xi:xi+config.patch['patchside']]
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                sc, res = gen.get_hidden_layer(patch)

            # Generate probability map
            sc = sc.array
            res = res.array
            if args.gpu >= 0:
                sc = chainer.backends.cuda.to_cpu(sc)
                res = chainer.backends.cuda.to_cpu(res)

            sc_map[:, zi:zi+config.patch['patchside'],yi:yi+config.patch['patchside'],xi:xi+config.patch['patchside']] += np.squeeze(sc)
            res_map[:, zi:zi+config.patch['patchside'],yi:yi+config.patch['patchside'],xi:xi+config.patch['patchside']] += np.squeeze(res)
            overlap_count[zi:zi+config.patch['patchside'],yi:yi+config.patch['patchside'],xi:xi+config.patch['patchside']] += 1

        print('     Save image')
        sc_map = sc_map[:, :ze,:ye,:xe]
        res_map = res_map[:, :ze,:ye,:xe]
        overlap_count = overlap_count[:ze,:ye,:xe]
        sc_map[:,...] /= overlap_count
        res_map[:,...] /= overlap_count

        # Save prediction map
        result_dir = os.path.join(args.base, args.out)
        sc_dir = '{}/sc'.format(result_dir)
        res_dir = '{}/res'.format(result_dir)
        if not os.path.exists(sc_dir):
            os.makedirs(sc_dir)
        if not os.path.exists(res_dir):
            os.makedirs(res_dir)

        sc_ims = []
        res_ims = []
        for loop in range(sc_map.shape[0]):
            fn = os.path.splitext(os.path.basename(i[0]))[0]

            scImage = sitk.GetImageFromArray(sc_map[loop,...])
            scImage.SetSpacing(sitkTri.GetSpacing())
            scImage.SetOrigin(sitkTri.GetOrigin())
            sitk.WriteImage(scImage, '{}/{}_{:04d}.mhd'.format(sc_dir, fn, loop))

            resImage = sitk.GetImageFromArray(res_map[loop,...])
            resImage.SetSpacing(sitkTri.GetSpacing())
            resImage.SetOrigin(sitkTri.GetOrigin())
            sitk.WriteImage(resImage, '{}/{}_{:04d}.mhd'.format(res_dir, fn, loop))

            # Rescale for visualization
            nd_sc = sitk.GetArrayFromImage(sitk.Cast(sitk.RescaleIntensity(scImage), sitk.sitkUInt8))[240,30:130,80:180]
            nd_res = sitk.GetArrayFromImage(sitk.Cast(sitk.RescaleIntensity(resImage), sitk.sitkUInt8))[240,30:130,80:180]
            sc_ims.append(nd_sc)
            res_ims.append(nd_res)

        sc_ims = np.asarray(sc_ims).transpose(1,0,2).reshape((100, -1))
        res_ims = np.asarray(res_ims).transpose(1,0,2).reshape((100, -1))
        ims = np.vstack((sc_ims, res_ims))
        preview_path = '{}/preview.png'.format(result_dir)
        figure = plt.figure()
        plt.imshow(ims, cmap='gray', interpolation='none')
        plt.axis('off')
        plt.savefig(preview_path, dpi=300)
        with open('{}/preview.pickle'.format(result_dir), 'wb') as f:
            pickle.dump(figure, f)

        plt.show()

    print(' Inference done')

if __name__ == '__main__':
    main()
