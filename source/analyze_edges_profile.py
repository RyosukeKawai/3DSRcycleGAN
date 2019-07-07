#coding:utf-8
import os, sys, time
import argparse, yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '..')))
import util.ioFunction_version_4_3 as IO

def extract_intensity_profile(img, mode, slice, fixed_pos, start, end, direction):
    """
    @param: img: ndarray image
    @param: mode: how to see.('axial', 'coronal', 'sagittal')
    @param: slice
    @param: fixed_pos
    @param: start
    @param: end
    @param: direction: horizontal or vertical
    """
    if not img.ndim == 3:
        print('Type Error. Please check img.ndim')
        raise

    mask = np.zeros(img.shape, dtype=np.uint8)

    if mode == 'axial':
        if direction == 'horizontal':
            profile = img[slice, fixed_pos, start:end]
            mask[slice, fixed_pos, start:end] = 255
            return profile, mask
        elif direction == 'vertical':
            profile = img[slice, start:end, fixed_pos]
            mask[slice, start:end, fixed_pos] = 255
            return profile, mask
        else:
            raise

    elif mode == 'coronal':
        if direction == 'horizontal':
            profile = img[fixed_pos, slice, start:end]
            mask[fixed_pos, slice, start:end] = 255
            return profile, mask
        elif direction == 'vertical':
            profile = img[start:end, slice, fixed_pos]
            mask[start:end, slice, fixed_pos] = 255
            return profile, mask
        else:
            raise

    elif mode == 'sagittal':
        if direction == 'horizontal':
            profile = img[fixed_pos, start:end, slice]
            mask[fixed_pos, start:end, slice] = 255
            return profile, mask
        elif direction == 'vertical':
            profile = img[start:end, fixed_pos, slice]
            mask[start:end, fixed_pos, slice] = 255
            return profile, mask
        else:
            raise

    else:
        raise NotImplementedError()


def main():
    parser = argparse.ArgumentParser(description='Analyze edge profile')
    parser.add_argument('--inputImageFile', '-i', help='input image file')
    parser.add_argument('--outputFile', '-o', help='output file')
    parser.add_argument('--makeMaskFlag', '-m', type=bool, default=False,
                        help='Whether make mask')

    parser.add_argument('--mode', type=str, default='axial',
                        help='Select slice plane(axial or sagittal or coronal)')
    parser.add_argument('--slice', type=int, help='Slice number')
    parser.add_argument('--fixed_pos', type=int, help='fixed positions')
    parser.add_argument('--start', type=int, help='start coordinate')
    parser.add_argument('--end', type=int, help='end coordinate')
    parser.add_argument('--direction', type=str, default='horizontal',
                        help='horizontal or vertical')
    args = parser.parse_args()

    # Read image
    img, dict = IO.read_mhd_and_raw_withoutSitk(args.inputImageFile)
    img = img.reshape((dict['DimSize'][2], dict['DimSize'][1], dict['DimSize'][0])).astype('float')

    profile, mask = extract_intensity_profile(img,
                                args.mode,
                                args.slice,
                                args.fixed_pos,
                                args.start, args.end,
                                args.direction)
    positions = np.arange(args.start, args.end)

    filename, _ = os.path.splitext(os.path.basename(args.outputFile))
    if args.makeMaskFlag:
        IO.write_mhd_and_raw_withoutSitk(mask.flatten(),
                                        '{}/{}_mask.mhd'.format(os.path.dirname(args.outputFile), filename),
                                        3,
                                        dict['DimSize'],
                                        dict['ElementSpacing'])

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(positions, profile, marker='x')
    plt.xticks(np.arange(args.start, args.end+1, 10))
    plt.xlabel('Position')
    plt.ylabel('Intensity')
    plt.title('{}-{}'.format(args.mode, args.direction))
    plt.savefig(args.outputFile)

    df = pd.DataFrame({'Position':positions, 'Intensity':profile})
    df.to_csv('{}/{}.csv'.format(os.path.dirname(args.outputFile), filename), index=False, encoding="utf-8", mode='w')

    with open('{}/{}_config.yml'.format(os.path.dirname(args.outputFile), filename), 'w') as f:
        f.write(yaml.dump(vars(args), default_flow_style=False))

if __name__ == '__main__':
    main()
