# coding:utf-8
import os, sys, time
import argparse, yaml, shutil, math
import numpy as np
import pandas as pd
import SimpleITK as sitk
import chainer
import chainer.functions as F
import utils.yaml_utils  as yaml_utils
from utils.evaluations import calc_mse, calc_psnr, calc_score_on_fft_domain, calc_ssim, calc_zncc
from interpolate_resnet import Generator



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                        help='base directory path of program files')
    parser.add_argument('--config_path', type=str, default='configs/training.yml',
                        help='path to config file')
    parser.add_argument('--out', '-o', default='results/inference',
                        help='Directory to output the result')

    parser.add_argument('--modeldir', '-m', default='',
                        help='Load model data(snapshot)')

    parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Root directory path of input image')

    parser.add_argument('--save_flag', type=bool, default=False,
                        help='Decision flag whether to save criteria image')
    parser.add_argument('--filename_arg', type=str, default='val_1784',
                        help='Which do you want to use val_list or test_list')

    args = parser.parse_args()

    config = yaml_utils.Config(yaml.load(open(os.path.join(args.base, args.config_path))))
    print('GPU: {}'.format(args.gpu))

    # -----------parameter set-------------------------
    LR_PATCH_SIDE, HR_PATCH_SIDE = config.patch['lr_patchside'], config.patch['patchside']
    LR_PATCH_SIZE, HR_PATCH_SIZE = int(LR_PATCH_SIDE ** 3), int(HR_PATCH_SIDE ** 3)
    print('HR PATCH SIZE: {}'.format(HR_PATCH_SIZE))
    print('LR PATCH SIZE: {}'.format(LR_PATCH_SIZE))

    print('')
    # -------------------------------------------------

    # -----------network set-------------------------
    gen = Generator()
    # ------------------------------------------------

    # -----------out dir set-----------------------------------------
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

    create_result_dir(args.base, args.out, args.config_path, config)
    # ------------------------------------------------------------------

    # -----------------read data ----------------------------------------
    path_pairs = []
    with open(os.path.join(args.base, config.dataset[args.filename_arg])) as paths_file:
        for line in paths_file:
            line = line.split()
            if not line: continue
            path_pairs.append(line[:])

    MIN_MIN, MAX_MAX = float(path_pairs[2][0]), float(path_pairs[2][1])

    print('    HR from: {}'.format(path_pairs[0][1]))
    print('MIN: {}'.format(MIN_MIN))
    print('MAX: {}'.format(MAX_MAX))
    sitkGt = sitk.ReadImage(os.path.join(args.root, path_pairs[0][1]))
    GT = sitk.GetArrayFromImage(sitkGt).astype(np.uint16)
    # -------------------------------------------------------------------


    for model_num in range(10000, 100000, 1000):

        # load model
        print('model_num = {}'.format((model_num)))
        model_path_num = '{}/gen_iter_{}.npz'.format(args.modeldir,model_num)
        model_path = os.path.join(args.base, model_path_num)
        chainer.serializers.load_npz(model_path, gen)
        if args.gpu >= 0:
            chainer.backends.cuda.set_max_workspace_size(1024 * 1024 * 1024)  # 1GB
            chainer.backends.cuda.get_device_from_id(args.gpu).use()
            gen.to_gpu()
        xp = gen.xp

        # -------read val_coordinate ----------------------------------------
        gt_val_coordinate_path = path_pairs[1][0]
        gt_coordinate = pd.read_csv(os.path.join(args.root, gt_val_coordinate_path),
                                    names=("x", "y", "z")).values.tolist()
        NUM_DATA = len(gt_coordinate)
        # -------------------------------------------------------------------

        psnr_total = 0.0

        print(' Inference start')
        for i in range(len(gt_coordinate)):
            if i%100==0:print('{}/{}'.format(i, NUM_DATA))
            x, y, z = gt_coordinate[i]

            # for PLR   extract from HR 128*12*128
            gt_plr = GT[z:z + 128, y:y + 128, x:x + 128]  # 128*128*128
            gt_plr.astype(np.float32)
            gt_plr = (gt_plr - MIN_MIN) / (MAX_MAX - MIN_MIN)# [0,1]

            # extract_HR_center_patch 64*64*64---------------------------------
            gt_patch = GT[z + config.patch['interval']:z + config.patch['interval'] + HR_PATCH_SIDE,
                       y + config.patch['interval']:y + config.patch['interval'] + HR_PATCH_SIDE,
                       x + config.patch['interval']:x + config.patch['interval'] + HR_PATCH_SIDE]
            gt_psnr = ((gt_patch - MIN_MIN) / (MAX_MAX - MIN_MIN)) * 255.  # float [0,255]

            # make PLR from gt_plr
            gt_plr = chainer.Variable(xp.array(gt_plr[np.newaxis, np.newaxis, :], dtype=xp.float32))
            PLR = F.average_pooling_3d(gt_plr, 8, 8, 0)

            # Calculate maximum of number of patch at each side
            ze = ye = xe = 128
            xm = int(math.ceil((float(xe) / float(config.patch['interval']))))-1
            ym = int(math.ceil((float(ye) / float(config.patch['interval']))))-1
            zm = int(math.ceil((float(ze) / float(config.patch['interval']))))-1
            edge = 0.

            # inferred_map = np.zeros((ze+HR_PATCH_SIDE,ye+HR_PATCH_SIDE, xe+HR_PATCH_SIDE))
            inferred_map = np.zeros((ze,ye, xe))
            overlap_count = np.zeros(inferred_map.shape)

            # Patch loop
            for s in range(xm * ym * zm):
                xi = int(s % xm) * config.patch['interval']
                yi = int((s % (ym * xm)) / xm) * config.patch['interval']
                zi = int(s / (ym * xm)) * config.patch['interval']

                lr_x = int((xi / 8.0) + 0.5)
                lr_y = int((yi / 8.0) + 0.5)
                lr_z = int((zi / 8.0) + 0.5)
                x_s, x_e = lr_x, lr_x + LR_PATCH_SIDE + edge
                y_s, y_e = lr_y, lr_y + LR_PATCH_SIDE + edge
                z_s, z_e = lr_z, lr_z + LR_PATCH_SIDE + edge

                # Extract patch from original image 8*8*8
                patch = PLR[:, :, z_s:z_e, y_s:y_e, x_s:x_e]

                with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                    inferred_patch = gen(patch)

                # Generate probability map
                inferred_patch = inferred_patch.data
                if args.gpu >= 0:
                    inferred_patch = chainer.backends.cuda.to_cpu(inferred_patch)

                inferred_map[zi:zi + HR_PATCH_SIDE, yi:yi + HR_PATCH_SIDE, xi:xi + HR_PATCH_SIDE] += np.squeeze(
                    inferred_patch)
                overlap_count[zi:zi + HR_PATCH_SIDE, yi:yi + HR_PATCH_SIDE, xi:xi + HR_PATCH_SIDE] += 1

            inferred_map = inferred_map[:ze, :ye, :xe]
            overlap_count = overlap_count[:ze, :ye, :xe]
            inferred_map /= overlap_count
            inferred_map_psnr = ((inferred_map + 1.) / 2.) * 255.  # for psnr range[0,255]

            inferred_map_psnr = inferred_map_psnr[config.patch['interval']:config.patch['interval'] + HR_PATCH_SIDE,
                                config.patch['interval']:config.patch['interval'] + HR_PATCH_SIDE,
                                config.patch['interval']:config.patch['interval'] + HR_PATCH_SIDE]


            psnr_total += calc_psnr(gt_psnr, inferred_map_psnr)

        out_path = args.out + '/{}'.format(model_num)
        result_dir = os.path.join(args.base, out_path)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        psnr_total = psnr_total / NUM_DATA


        df = pd.DataFrame({'PSNR': [psnr_total]})
        df.to_csv('{}/results.csv'.format(result_dir), index=False, encoding='utf-8', mode='a')

if __name__ == '__main__':
    main()
