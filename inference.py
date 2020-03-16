#coding:utf-8
import os, sys, time
import argparse, yaml, shutil, math
import numpy as np
import pandas as pd
import SimpleITK as sitk
import chainer

import utils.ioFunctions as IO
import utils.yaml_utils  as yaml_utils
from utils.evaluations import calc_mse, calc_psnr, calc_score_on_fft_domain, calc_ssim, calc_zncc
from model import Generator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                        help='base directory path of program files')
    parser.add_argument('--config_path', type=str, default='configs/training.yml',
                        help='path to config file')
    parser.add_argument('--out', '-o', default= 'results/inference',
                        help='Directory to output the result')

    parser.add_argument('--model', '-m', default='',
                        help='Load model data(snapshot)')

    parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Root directory path of input image')

    parser.add_argument('--save_flag', type=bool, default=False,
                        help='Decision flag whether to save criteria image')
    parser.add_argument('--filename_arg', type=str, default='val_fn',
                        help='Which do you want to use val_list or test_list')

    args = parser.parse_args()

    config = yaml_utils.Config(yaml.load(open(os.path.join(args.base, args.config_path))))
    print('GPU: {}'.format(args.gpu))
    UPSAMPLING_RATE = config.patch['upsampling_rate']
    LR_PATCH_SIDE, HR_PATCH_SIDE = config.patch['lr_patchside'], config.patch['patchside']
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
            os.path.join(base_dir, config.network['fn']), result_dir)
        copy_to_result_dir(
            os.path.join(base_dir, config.updater['fn']), result_dir)
        copy_to_result_dir(
            os.path.join(base_dir, config.dataset[args.filename_arg]), result_dir)

    create_result_dir(args.base,  args.out, args.config_path, config)

    # Read data
    path_pairs = []
    with open(os.path.join(args.base, config.dataset[args.filename_arg])) as paths_file:
        for line in paths_file:
            line = line.split()
            if not line : continue
            path_pairs.append(line[:])

    print(' Inference start')
    for i in path_pairs:
        print('    LR from: {}'.format(i[0]))
        print('    HR from: {}'.format(i[1]))
        #Read data and reshape
        sitkLr = sitk.ReadImage(os.path.join(args.root, i[0]))
        lr = sitk.GetArrayFromImage(sitkLr).astype("float32")
        sitkGt = sitk.ReadImage(os.path.join(args.root, i[1]))
        gt = sitk.GetArrayFromImage(sitkGt).astype("float")
        lr = (lr-LR_MIN)/ (LR_MAX-LR_MIN)

        # Calculate maximum of number of patch at each side
        ze, ye, xe = gt.shape
        xm = int(math.ceil((float(xe)/float(config.patch['interval']))))
        ym = int(math.ceil((float(ye)/float(config.patch['interval']))))
        zm = int(math.ceil((float(ze)/float(config.patch['interval']))))

        margin = ((0, LR_PATCH_SIDE),
                  (0, LR_PATCH_SIDE),
                  (0, LR_PATCH_SIDE))
        lr = np.pad(lr, margin, 'edge')
        lr = chainer.Variable(xp.array(lr[np.newaxis, np.newaxis, :], dtype=xp.float32))

        inferred_map = np.zeros((ze+HR_PATCH_SIDE,ye+HR_PATCH_SIDE, xe+HR_PATCH_SIDE))
        overlap_count = np.zeros(inferred_map.shape)

        # Patch loop
        print('     #Patches {}'.format(xm*ym*zm))
        for s in range(xm*ym*zm):
            xi = int(s%xm)*config.patch['interval']
            yi = int((s%(ym*xm))/xm)*config.patch['interval']
            zi = int(s/(ym*xm))*config.patch['interval']

            lr_x = int(xi/UPSAMPLING_RATE+0.5)
            lr_y = int(yi/UPSAMPLING_RATE+0.5)
            lr_z = int(zi/UPSAMPLING_RATE+0.5)
            x_s, x_e = lr_x, lr_x + LR_PATCH_SIDE
            y_s, y_e = lr_y, lr_y + LR_PATCH_SIDE
            z_s, z_e = lr_z, lr_z + LR_PATCH_SIDE
            # Extract patch from original image
            patch = lr[:,:,z_s:z_e,y_s:y_e,x_s:x_e]
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                inferred_patch = gen(patch)

            # Generate probability map
            inferred_patch = inferred_patch.data
            if args.gpu >= 0:
                inferred_patch = chainer.backends.cuda.to_cpu(inferred_patch)

            inferred_map[zi:zi+HR_PATCH_SIDE,yi:yi+HR_PATCH_SIDE,xi:xi+HR_PATCH_SIDE] += np.squeeze(inferred_patch)
            overlap_count[zi:zi+HR_PATCH_SIDE,yi:yi+HR_PATCH_SIDE,xi:xi+HR_PATCH_SIDE] += 1

        print('     Save image')
        inferred_map = inferred_map[:ze,:ye,:xe]
        overlap_count = overlap_count[:ze,:ye,:xe]
        inferred_map /= overlap_count
        inferred_map = ((inferred_map+1)/2*255)

        # Save prediction map
        inferenceImage = sitk.GetImageFromArray(inferred_map)
        inferenceImage.SetSpacing(sitkGt.GetSpacing())
        inferenceImage.SetOrigin(sitkGt.GetOrigin())
        result_dir = os.path.join(args.base, args.out)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        fn = os.path.splitext(os.path.basename(i[0]))[0]
        sitk.WriteImage(inferenceImage, '{}/{}.mhd'.format(result_dir, fn))

        print('     Start evaluations ')
        start = time.time()
        mse_const = calc_mse(gt, inferred_map)
        psnr_const = calc_psnr(gt, inferred_map)
        ssim_const = calc_ssim(gt, inferred_map)
        zncc_const = calc_zncc(gt, inferred_map)
        diff_spe_const, diff_spe, diff_ang_const, diff_ang = calc_score_on_fft_domain(gt, inferred_map)
        df = pd.DataFrame({'MSE': [mse_const], 'PSNR':[psnr_const], 'SSIM':[ssim_const], 'ZNCC':[zncc_const], 'MAE-Power':[diff_spe_const], 'MAE-Angle':[diff_ang_const]})
        df.to_csv('{}/results.csv'.format(result_dir), index=False, encoding='utf-8', mode='w')
        print('     Finsish evaluations: {:.3f} [s]'.format(time.time() - start))
        if args.save_flag:
            def array2sitk(arr, spacing=[], origin=[]):
                if not len(spacing) == arr.ndim or len(origin) == arr.ndim:
                    print("Dimension Error")
                    quit()

                sitkImg = sitk.GetImageFromArray(arr)
                sitkImg.SetSpacing(spacing)
                sitkImg.SetOrigin(origin)
                return sitkImg

            diffSpeImage = array2sitk(diff_spe, [1,1,1], [0,0,0])
            diffAngImage = array2sitk(diff_ang, [1,1,1], [0,0,0])
            sitk.WriteImage(diffSpeImage, '{}/{}-diff_power_spe.mhd'.format(result_dir, fn))
            sitk.WriteImage(diffAngImage, '{}/{}-diff_angle.mhd'.format(result_dir, fn))

    print(' Inference done')


if __name__ == '__main__':
    main()