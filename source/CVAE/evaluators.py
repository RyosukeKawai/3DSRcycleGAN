#coding:utf-8

"""
@auther tozawa
@date 20181213
"""
import os, sys, time, copy
import random
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import chainer.distributions as D
from chainer import reporter
import itertools
import SimpleITK as sitk

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def generate_samples(model, out_dir, rows=5, cols=5, n_latent=64):
    xp = model.xp
    n_latent = xp.random.rand(rows*cols*n_latent, dtype=xp.float32).reshape(rows*cols, n_latent)
    @chainer.training.make_extension()
    def evaluation(trainer):
        decoder = model.decoder

        # Generate samples
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            y = decoder(n_latent)
        y = chainer.backends.cuda.to_cpu(y.array)

        y = adjust_array_for_output(y, rows, cols)

        preview_dir = '{}/preview'.format(out_dir)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        preview_path = '{}/preview_{}.png'.format(preview_dir, trainer.updater.iteration)
        sitkImg = sitk.GetImageFromArray(y)
        sitk.WriteImage(sitkImg, preview_path)

    return evaluation

def projection_to_latent_space(model, out_dir, dataset, n_images=500, batchsize=10):
    idx = random.sample(range(0, dataset.__len__()), k=n_images)
    patches = np.asarray(dataset[idx], dtype=np.float32)
    @chainer.training.make_extension()
    def evaluation(trainer):
        encoder = model.encoder
        xp = model.xp

        latent_list = []
        for n in range(0, n_images-batchsize+1, batchsize):
            patch = chainer.Variable(xp.array(patches[n:n+batchsize, ...], dtype=xp.float32))
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                mu, _ = encoder(patch)
            mu = chainer.backends.cuda.to_cpu(mu.array)
            #mu = np.squeeze(mu, axis=0)
            for n in mu:
                latent_list.append(n)

        latent_list = np.asarray(latent_list)
        preview_dir = '{}/preview'.format(out_dir)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        plot(latent_list, preview_dir, trainer.updater.iteration)
        #preview_path = '{}/preview_latent_{}.png'.format(preview_dir, trainer.updater.iteration)

    return evaluation

def reconstruction_img(model, out_dir, dataset):
    idx=random.sample(range(0, dataset.__len__()), k=9)
    @chainer.training.make_extension()
    def evaluation(trainer):
        encoder = model.encoder
        decoder = model.decoder
        xp = model.xp

        x = chainer.Variable(xp.asarray(dataset[idx], dtype=xp.float32))

        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            mu, _ = encoder(x)
            y = decoder(mu)
        y = chainer.backends.cuda.to_cpu(y.array)
        x = chainer.backends.cuda.to_cpu(x.array)
        rows = cols = 3
        y = adjust_array_for_output(y, rows, cols)
        x = adjust_array_for_output(x, rows, cols)

        preview_dir = '{}/preview'.format(out_dir)
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)
        preview_path = '{}/reconstr_{}.png'.format(preview_dir, trainer.updater.iteration)
        sitkImg = sitk.GetImageFromArray(y)
        sitk.WriteImage(sitkImg, preview_path)

        preview_path = '{}/org_{}.png'.format(preview_dir, trainer.updater.iteration)
        sitkImg = sitk.GetImageFromArray(x)
        sitk.WriteImage(sitkImg, preview_path)

    return evaluation


def plot(data, result_dir, number, label='training'):
    if not data.shape[1] == 3:
        raise NotImplementedError()

    color = 'navy'
    alpha = 0.5
    figure = plt.figure()
    ax = figure.add_subplot(111, aspect='equal', projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], lw=0, c=color, alpha=alpha, label=label)
    ax.axis('tight')
    ax.legend(loc='best', shadow=False, scatterpoints=1)
    plt.savefig('{}/plot_{}.png'.format(result_dir, number), dpi=120)
    # with open('{}/plot_{}.pickle'.format(result_dir, number), 'wb') as f:
    #     pickle.dump(figure, f)
    #plt.show()
    plt.close()

class CvaeEvaluator(chainer.training.extensions.Evaluator):
    def __init__(self, iterator, model,
                converter=chainer.dataset.concat_examples,
                device=None, eval_hook=None):
        if isinstance(iterator, chainer.dataset.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        self._targets = {'model':model}
        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook

    def evaluate(self):
        iterator = self._iterators['main']
        model = self._targets['model']
        encoder = model.encoder
        decoder = model.decoder
        k = model.k
        beta = model.beta

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            it = copy.copy(iterator)

        summary = reporter.DictSummary()

        for batch in it:
            observation = {}
            with reporter.report_scope(observation):
                x = self.converter(batch, self.device)
                with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                    mu, ln_var = encoder(x)
                    batchsize = len(mu.array)
                    kl_penalty = F.gaussian_kl_divergence(mean=mu, ln_var=ln_var) / batchsize
                    reconstr = 0
                    for l in range(k):
                        z = F.gaussian(mu, ln_var)
                        recon = decoder(z)
                        reconstr += 0.5 * F.mean_squared_error(recon, x) * x.shape[2] * x.shape[3] * x.shape[4] / k

                    loss = (reconstr + beta * kl_penalty)

                observation['validation/loss'] = loss
                observation['validation/reconstr'] = reconstr
                observation['validation/kl_penalty'] = kl_penalty

            summary.add(observation)

        return summary.compute_mean()

def adjust_array_for_output(nda, rows, cols):
    x = np.asarray(np.clip((nda+1.)*127.5, 0.0, 255.0), dtype=np.uint8)
    _, _, D, H, W = x.shape
    x = x[:,:,D//2,:,:]
    x = x.reshape((rows, cols, 1, H, W))
    x = x.transpose((0,3,1,4,2))
    x = x.reshape((rows*H, cols*W, 1))
    return x[:,:,0]
