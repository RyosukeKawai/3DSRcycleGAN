#coding:utf-8
import os, time, sys, random
import argparse, yaml, shutil, math
import numpy as np
import pandas as pd
import chainer
import chainer.links as L
import chainer.functions as F
import SimpleITK as sitk

import utils.ioFunctions as IO
from utils.evaluations import calc_mse, calc_psnr, calc_score_on_fft_domain, calc_ssim, calc_zncc

class SRResNetEvaluator(chainer.training.extensions.Evaluator):
    def __init__(self, iterator, gen,
                    converter=chainer.dataset.concat_examples,
                    device=None, eval_hook=None):
        if isinstance(iterator, chainer.dataset.Iterator):
            iterator = {"main":iterator}
        self._iterators = iterator
        self._targets = {"gen" : gen}
        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook

    def evaluate(self):
        iterator = self._iterators["main"]
        gen = self._targets["gen"]

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator) #shallow copy

        summary = reporter.DictSummary()

        for batch in it:
            observation ={}
            with reporter.report_scope(observation):
                x, t = self.converter(batch, self.device)
                with chainer.using_config("train", False), chainer.using_config('enable_backprop', False):
                        y = gen(x)
                loss = F.mean_squared_error(y, t)

                observation["gen/val/loss"] = loss

            summary.add(observation)

        return summary.compute_mean()

class PsnrEvaluator(chainer.training.extensions.Evaluator):
    def __init__(self, iterator, gen, data_range,
                    converter=chainer.dataset.concat_examples,
                    device=None, eval_hook=None):
        if isinstance(iterator, chainer.dataset.Iterator):
            iterator = {"main":iterator}
        self._iterators = iterator
        self._targets = {"gen" : gen}
        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook
        self._data_range = data_range

    def evaluate(self):
        iterator = self._iterators["main"]
        gen = self._targets["gen"]

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator) #shallow copy

        summary = reporter.DictSummary()

        for batch in it:
            observation ={}
            with reporter.report_scope(observation):
                x, t = self.converter(batch, self.device)
                with chainer.using_config("train", False), chainer.using_config('enable_backprop', False):
                        y = gen(x)
                mse = F.mean_squared_error(y, t)
                psnr = 10 * F.log10(self._data_range/mse)

                observation["gen/val/psnr"] = psnr

            summary.add(observation)

        return summary.compute_mean()

def reconstruct_hr_img(gen, out_dir, dataset, rows=5, cols=5, converter=chainer.dataset.concat_examples):
    idx = random.sample(range(0, dataset.__len__()), k=rows*cols)
    patches, hr_img = converter(dataset[idx], device=-1)
    @chainer.training.make_extension()



    def evaluation(trainer):

        xp = gen.xp
        x = chainer.Variable(xp.array(patches, dtype=xp.float32))
        hr = hr_img
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            y = gen(x)

        y = chainer.backends.cuda.to_cpu(y.array)
        x = chainer.backends.cuda.to_cpu(x.array)

        y = adjust_array_for_output(y, rows, cols, range_type='0')
        x = adjust_array_for_output(x, rows, cols, range_type='1')
        hr = adjust_array_for_output(hr, rows, cols, range_type='0')

        preview_dir = '{}/preview'.format(out_dir)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)

        preview_path = '{}/reconstr_{}.png'.format(preview_dir, trainer.updater.iteration)
        sitkImg = sitk.GetImageFromArray(y)
        sitk.WriteImage(sitkImg, preview_path)

        preview_path = '{}/org_{}.png'.format(preview_dir, trainer.updater.iteration)
        sitkImg = sitk.GetImageFromArray(x)
        sitk.WriteImage(sitkImg, preview_path)

        preview_path = '{}/gt_{}.png'.format(preview_dir, trainer.updater.iteration)
        sitkImg = sitk.GetImageFromArray(hr)
        sitk.WriteImage(sitkImg, preview_path)

    return evaluation

def segment_hr_img(dis, out_dir, dataset, rows=5, cols=5, converter=chainer.dataset.concat_examples):
    idx = random.sample(range(0, dataset.__len__()), k=rows*cols)
    _, patches, gt = converter(dataset[idx], device=-1)
    @chainer.training.make_extension()
    def evaluation(trainer):
        xp = dis.xp

        x = chainer.Variable(xp.array(patches, dtype=xp.float32))
        labels = gt
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            _, y = dis(x)
            y = F.sigmoid(y)

        y = chainer.backends.cuda.to_cpu(y.array)
        x = chainer.backends.cuda.to_cpu(x.array)

        labels = labels[:, 1, :, :, :]
        labels = labels[:, np.newaxis, :, :, :]

        y = adjust_array_for_output(y, rows, cols, range_type='1')
        x = adjust_array_for_output(x, rows, cols, range_type='0')
        labels = adjust_array_for_output(labels, rows, cols, range_type='1')

        preview_dir = '{}/preview'.format(out_dir)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)

        preview_path = '{}/predict_{}.png'.format(preview_dir, trainer.updater.iteration)
        sitkImg = sitk.GetImageFromArray(y)
        sitk.WriteImage(sitkImg, preview_path)

        preview_path = '{}/input_{}.png'.format(preview_dir, trainer.updater.iteration)
        sitkImg = sitk.GetImageFromArray(x)
        sitk.WriteImage(sitkImg, preview_path)

        preview_path = '{}/gt_label_{}.png'.format(preview_dir, trainer.updater.iteration)
        sitkImg = sitk.GetImageFromArray(labels)
        sitk.WriteImage(sitkImg, preview_path)

    return evaluation

def adjust_array_for_output(nda, rows, cols, range_type='0'):
    if range_type.lower() == '0':# nda range is -1 to 1
        x = np.asarray(np.clip((nda+1.)*127.5, 0., 255.), dtype=np.uint8)
    elif range_type.lower() == '1': # nda range is 0 to 1
        x = np.asarray(np.clip((nda*255.), 0., 255.), dtype=np.uint8)
    else:
        raise NotImplementedError()


    _, _, D, H, W = x.shape
    x = x[:,:,D//2,:,:]
    x = x.reshape((rows, cols, 1, H, W))
    x = x.transpose((0,3,1,4,2))
    x = x.reshape((rows*H, cols*W, 1))
    return x[:,:,0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputImage', '-i',
                        help='Directory to input image path')
    parser.add_argument('--groundTruthImage', '-g',
                        help='Directory to ground truth image path')
    parser.add_argument('--outputImage', '-o',
                        help='Directory to output image path')
    parser.add_argument('--save_flag', type=bool, default=False,
                        help='Decision flag whether to save criteria image')
    args = parser.parse_args()

    print('     Start evaluations ')
    sitkGt = sitk.ReadImage(args.groundTruthImage)
    gt = sitk.GetArrayFromImage(sitkGt).astype("float")
    sitkImg = sitk.ReadImage(args.inputImage)
    img = sitk.GetArrayFromImage(sitkImg).astype("float")

    start = time.time()
    mse_const = calc_mse(gt, img)
    psnr_const = calc_psnr(gt, img)
    ssim_const = calc_ssim(gt, img)
    zncc_const = calc_zncc(gt, img)

    diff_spe_const, diff_spe, diff_ang_const, diff_ang = calc_score_on_fft_domain(gt, img)
    df = pd.DataFrame({'MSE': [mse_const], 'PSNR':[psnr_const], 'SSIM':[ssim_const], 'ZNCC':[zncc_const], 'MAE-Power':[diff_spe_const], 'MAE-Angle':[diff_ang_const]})

    result_dir = os.path.dirname(args.outputImage)

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
