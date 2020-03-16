# coding:utf-8
import os, time, sys, random
import numpy as np
import chainer,copy
import chainer.links as L
import chainer.functions as F
import SimpleITK as sitk


class SRResNetEvaluator(chainer.training.extensions.Evaluator):
    def __init__(self, iterator, gen,
                 converter=chainer.dataset.concat_examples,
                 device=None, eval_hook=None):
        if isinstance(iterator, chainer.dataset.Iterator):
            iterator = {"main": iterator}
        self._iterators = iterator
        self._targets = {"gen": gen}
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
            it = copy.copy(iterator)  # shallow copy

        summary = chainer.reporter.DictSummary()

        for batch in it:
            observation = {}
            with chainer.reporter.report_scope(observation):
                x, t = self.converter(batch, self.device)
                with chainer.using_config("train", False), chainer.using_config('enable_backprop', False):
                    y = gen(x)
                loss = F.mean_squared_error(y, t)

                observation["gen/val/loss"] = loss

            summary.add(observation)

        return summary.compute_mean()


def reconstruct_hr_img(gen,out_dir, dataset, rows=5, cols=5, converter=chainer.dataset.concat_examples):
    idx = random.sample(range(0, dataset.__len__()), k=rows * cols)
    patches, patches2 = converter(dataset[idx], device=-1)

    @chainer.training.make_extension()
    def evaluation(trainer):
        xp = gen.xp
        LR = chainer.Variable(xp.array(patches, dtype=xp.float32))
        HR = chainer.Variable(xp.array(patches2, dtype=xp.float32))

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
           y = gen(LR)#LR-HR
           z = F.average_pooling_3d(y,8,8,0)#LR-HR-LR
           a=F.average_pooling_3d(HR,8,8,0)#HR-LR
           b=gen(a)#HR-LR-HR
           

        y = chainer.backends.cuda.to_cpu(y.array)
        LR = chainer.backends.cuda.to_cpu(LR.array)
        z = chainer.backends.cuda.to_cpu(z.array)

        a = chainer.backends.cuda.to_cpu(a.array)
        HR = chainer.backends.cuda.to_cpu(HR.array)
        b = chainer.backends.cuda.to_cpu(b.array)


        y = adjust_array_for_output(y, rows, cols, range_type='0')
        LR = adjust_array_for_output(LR, rows, cols, range_type='0')
        z = adjust_array_for_output(z, rows, cols, range_type='0')

        a = adjust_array_for_output(a, rows, cols, range_type='0')
        HR = adjust_array_for_output(HR, rows, cols, range_type='0')
        b = adjust_array_for_output(b, rows, cols, range_type='0')



        preview_dir = '{}/preview'.format(out_dir)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)

        preview_path = '{}/LR_HR_{}.png'.format(preview_dir, trainer.updater.iteration)
        sitkImg = sitk.GetImageFromArray(y)
        sitk.WriteImage(sitkImg, preview_path)

        preview_path = '{}/LR_HR_LR{}.png'.format(preview_dir, trainer.updater.iteration)
        sitkImg = sitk.GetImageFromArray(z)
        sitk.WriteImage(sitkImg, preview_path)

        preview_path = '{}/org_LR_{}.png'.format(preview_dir, trainer.updater.iteration)
        sitkImg = sitk.GetImageFromArray(LR)
        sitk.WriteImage(sitkImg, preview_path)

        preview_path = '{}/HR_LR_{}.png'.format(preview_dir, trainer.updater.iteration)
        sitkImg = sitk.GetImageFromArray(a)
        sitk.WriteImage(sitkImg, preview_path)

        preview_path = '{}/HR_LR_HR_{}.png'.format(preview_dir, trainer.updater.iteration)
        sitkImg = sitk.GetImageFromArray(b)
        sitk.WriteImage(sitkImg, preview_path)

        preview_path = '{}/org_HR_{}.png'.format(preview_dir, trainer.updater.iteration)
        sitkImg = sitk.GetImageFromArray(HR)
        sitk.WriteImage(sitkImg, preview_path)
    return evaluation


def adjust_array_for_output(nda, rows, cols, range_type='0'):
    if range_type.lower() == '0':  # nda range is -1 to 1
        x = np.asarray(np.clip((nda + 1.) * 127.5, 0.0, 255.0), dtype=np.uint8)
    elif range_type.lower() == '1':  # nda range is 0 to 1
        x = np.asarray(np.clip(nda * 255.0, 0.0, 255.0), dtype=np.uint8)
    else:
        raise NotImplementedError()
    _, _, D, H, W = x.shape
    x = x[:, :, D // 2, :, :]
    x = x.reshape((rows, cols, 1, H, W))
    x = x.transpose((0, 3, 1, 4, 2))
    x = x.reshape((rows * H, cols * W, 1))
    return x[:, :, 0]
