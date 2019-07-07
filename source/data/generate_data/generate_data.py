#coding:utf-8
import time, sys, os
import argparse
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '../..')))
import util.ioFunction_version_4_3 as IO

def main():
    parser = argparse.ArgumentParser(description='In ClinicalCT, To separate image for test, validation, and training')
    parser.add_argument('--inputImageFile', '-i', help='input image file')
    parser.add_argument('--outputDir', '-o', help='output Directory')
    parser.add_argument('--baseName', default='ClinicalCT.mhd')
    args = parser.parse_args()

    img, dict = IO.read_mhd_and_raw_withoutSitk(args.inputImageFile)
    img = img.reshape((dict['DimSize'][2], dict['DimSize'][1], dict['DimSize'][0]))
    xspacing, yspacing, zspacing = dict['ElementSpacing']
    XM_SPACING, YM_SPACING, ZM_SPACING = 0.07, 0.066, 0.07

    # For training
    XLIM = int(((1240 - 1) * XM_SPACING)/xspacing + 0.5)
    output = img[:,:, :XLIM]
    result_dir = '{}/train'.format(args.outputDir)
    os.makedirs(result_dir, exist_ok=True)
    IO.write_mhd_and_raw_withoutSitk(output.flatten(), '{}/{}'.format(result_dir, args.baseName), 3,
                                        [XLIM, dict['DimSize'][1], dict['DimSize'][2]], dict['ElementSpacing'])

    # For validation
    XS, XE = int(((1240) * XM_SPACING)/xspacing + 0.5), int(((1880 - 1) * XM_SPACING)/xspacing + 0.5)
    YS, YE = int(((820) * YM_SPACING)/yspacing + 0.5), int(((980 - 1) * YM_SPACING)/yspacing + 0.5)
    ZS, ZE = int(((770) * ZM_SPACING)/zspacing + 0.5), int(((1250 - 1) * ZM_SPACING)/zspacing + 0.5)
    output = img[ZS:ZE, YS:YE, XS:XE]
    result_dir = '{}/validation'.format(args.outputDir)
    os.makedirs(result_dir, exist_ok=True)
    IO.write_mhd_and_raw_withoutSitk(output.flatten(), '{}/{}'.format(result_dir, args.baseName), 3,
                                    [XE-XS, YE-YS, ZE-ZS], dict['ElementSpacing'])

    # For test
    XS, XE = int(((1240) * XM_SPACING)/xspacing + 0.5), int(((1880 - 1) * XM_SPACING)/xspacing + 0.5)
    YS, YE = int(((820) * YM_SPACING)/yspacing + 0.5), int(((980 - 1) * YM_SPACING)/yspacing + 0.5)
    ZS, ZE = int(((1250) * ZM_SPACING)/zspacing + 0.5), int(((1730 - 1) * ZM_SPACING)/zspacing + 0.5)
    output = img[ZS:ZE, YS:YE, XS:XE]
    result_dir = '{}/test'.format(args.outputDir)
    os.makedirs(result_dir, exist_ok=True)
    IO.write_mhd_and_raw_withoutSitk(output.flatten(), '{}/{}'.format(result_dir, args.baseName), 3,
                                    [XE-XS, YE-YS, ZE-ZS], dict['ElementSpacing'])


if __name__ == '__main__':
    main()
