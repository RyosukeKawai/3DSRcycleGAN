# coding:utf-8
import os, sys, time, random, yaml
import argparse, pickle
import numpy as np
import pandas as pd
import SimpleITK as sitk

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '../..')))
import util.ioFunction_version_4_3 as IO

def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--inputImageFile', '-i', help='input image file')
    parser.add_argument('--outputDir', '-o', help='output directory')
    parser.add_argument('--patch_coordinate_list', '-p', help='csv file that include patch origin')

    parser.add_argument('--num_patches', type=int, default=500,
                        help='number of patches that you want to extract')
    parser.add_argument('--patch_side',  type=int, default=64, help='patch side')
    parser.add_argument('--seed', type=int, default=0, help='seed for random')

    args = parser.parse_args()

    reset_seed(args.seed)

    print('----- Save configs -----')
    fn = os.path.splitext(os.path.basename(args.inputImageFile))[0]
    result_dir = args.outputDir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    with open('{}/{}_configs.yml'.format(result_dir, fn), 'w') as f:
        f.write(yaml.dump(vars(args), default_flow_style=False))

    print('----- Read dataset -----')
    coordinate_list = pd.read_csv(args.patch_coordinate_list, names=("x","y","z")).values.tolist()
    nda = IO.read_mhd_and_raw(args.inputImageFile)

    print('----- Extract patches ------')
    save_list = []
    for n in range(args.num_patches):
        idx = np.random.randint(0, len(coordinate_list))
        x, y, z =  coordinate_list[idx]
        save_list.append(coordinate_list[idx])
        x_s, x_e = x, x + args.patch_side
        y_s, y_e = y, y + args.patch_side
        z_s, z_e = z, z + args.patch_side
        patch = nda[z_s:z_e, y_s:y_e, x_s:x_e]

        sitkImg = sitk.GetImageFromArray(patch)
        sitkImg.SetSpacing([1,1,1])
        sitk.WriteImage(sitkImg, '{}/{}_{:04d}.mhd'.format(result_dir, fn, n))

    np.savetxt('{}/coordinate.csv'.format(result_dir), np.asarray(save_list, dtype=int), delimiter=',')


if __name__ == '__main__':
    main()
