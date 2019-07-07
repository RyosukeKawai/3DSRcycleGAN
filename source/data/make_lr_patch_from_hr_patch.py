#coding:utf-8
import os, sys, time
import argparse, glob, yaml
import SimpleITK as sitk
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '../..')))
import util.ioFunction_version_4_3 as IO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputImageFile', '-i', help='input image file')
    parser.add_argument('--outputDir', '-o', help='output directory')
    parser.add_argument('--upsampingRate', type=float, default=8, help='upsamping rate')
    args = parser.parse_args()

    hr_img = IO.read_mhd_and_raw(args.inputImageFile, False)
    hr_img.SetOrigin([0,0,0])
    hr_size = hr_img.GetSize()
    hr_spacing = hr_img.GetSpacing()
    new_spacing = [i*args.upsampingRate for i in hr_spacing]
    new_size = [int(hr_size[0]*(hr_spacing[0]/new_spacing[0])+0.5),
                int(hr_size[1]*(hr_spacing[1]/new_spacing[1])+0.5),
                int(hr_size[2]*(hr_spacing[2]/new_spacing[2])+0.5)]

    resampleFilter = sitk.ResampleImageFilter()

    lr_img = resampleFilter.Execute(hr_img,
                                    new_size,
                                    sitk.Transform(),
                                    sitk.sitkBSpline,
                                    hr_img.GetOrigin(),
                                    new_spacing,
                                    hr_img.GetDirection(), 0, hr_img.GetPixelID())

    # Save HR and LR images
    result_dir = args.outputDir
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    fn = os.path.splitext(os.path.basename(args.inputImageFile))[0]
    sitk.WriteImage(lr_img, '{}/{}.mhd'.format(result_dir, fn))

if __name__ == '__main__':
    main()
