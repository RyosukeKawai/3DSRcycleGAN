# coding: utf-8
import os, sys, time, random
import numpy as np
import argparse, pickle, yaml, shutil
import chainer
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from model import *
from dataset import CvaeDataset
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '../..')))
import util.yaml_utils  as yaml_utils

def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.backends.cuda.available:
        chainer.backends.cuda.cupy.random.seed(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                        help='base directory path of program files')
    parser.add_argument('--config_path', type=str, default='configs/training.yml',
                        help='path to config file')
    parser.add_argument('--out', '-o', default= 'results/projection',
                        help='Directory to output the result')

    parser.add_argument('--model', '-m', default='',
                        help='Load model data(snapshot)')

    parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Root directory path of input image')

    parser.add_argument('--num_patches', type=int, default=500,
                        help='number of patches that you want to extract')
    parser.add_argument('--filename', type=str, default='training_fn',
                        help='Which do you want to use val_fn or test_fn')

    parser.add_argument('--dataset', help='path to dataset pickle')
    args = parser.parse_args()

    if args.dataset:
        with open(args.dataset, 'rb') as f:
            dataset = pickle.load(f)
        out = '{}/{}'.format(args.out, args.filename)
        output_dir = os.path.join(args.base, out)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plot(dataset, output_dir)
        return

    print('----- Read configs ------')
    config = yaml_utils.Config(yaml.load(open(os.path.join(args.base, args.config_path))))
    print('GPU: {}'.format(args.gpu))

    print('----- Load model ------')
    encoder = Encoder()
    chainer.serializers.load_npz(args.model, encoder)
    if args.gpu >= 0:
        chainer.backends.cuda.set_max_workspace_size(1024 * 1024 * 1024) # 1GB
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        encoder.to_gpu()
    xp = encoder.xp

    print('----- Save configs ------')
    def create_result_dir(base_dir, output_dir, config_path, config):
        """https://github.com/pfnet-research/sngan_projection/blob/master/train.py"""
        result_dir = os.path.join(base_dir, output_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        if not os.path.exists('{}/model'.format(result_dir)):
            os.makedirs('{}/model'.format(result_dir))

        def copy_to_result_dir(fn, result_dir):
            bfn = os.path.basename(fn)
            shutil.copy(fn, '{}/{}'.format(result_dir, bfn))

        copy_to_result_dir(
            os.path.join(base_dir, config_path), result_dir)

        copy_to_result_dir(
            os.path.join(base_dir, config.network['fn']), result_dir)
        copy_to_result_dir(
            os.path.join(base_dir, config.dataset['val_fn']), result_dir)
        copy_to_result_dir(
            os.path.join(base_dir, config.dataset['training_fn']), result_dir)
        copy_to_result_dir(
            os.path.join(base_dir, args.model), '{}/model'.format(result_dir))

    out = '{}/{}'.format(args.out, args.filename)
    create_result_dir(args.base,  out, args.config_path, config)
    output_dir = os.path.join(args.base, out)

    print('----- Load dataset -----')
    dataset = CvaeDataset(args.root,
                        os.path.join(args.base, config.dataset[args.filename]),
                        config.patch['patchside'],
                        [config.patch['lrmin'], config.patch['lrmax']],
                        augmentation=False)

    print('----- Start projection ------')
    index_list = []
    latent_list = []
    for n in range(args.num_patches):
        idx = np.random.randint(0, dataset.__len__())
        patch = dataset.get_example(idx)
        patch = chainer.Variable(xp.array(patch[np.newaxis, :], dtype=xp.float32))
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            mu, _ = encoder(patch)

        mu = mu.array
        if args.gpu >= 0:
            mu = chainer.backends.cuda.to_cpu(mu)
        mu = np.squeeze(mu, axis=0)

        latent_list.append(mu)
        index_list.append(idx)

    latent_list = np.asarray(latent_list)
    print('----- Start save -----')
    np.savetxt('{}/coordinate_idx.csv'.format(output_dir), np.asarray(index_list, dtype=int), delimiter=',')
    np.savetxt('{}/latent_data.csv'.format(output_dir), latent_list, delimiter=',')
    with open('{}/latent_data.pickle'.format(output_dir), 'wb') as f:
        pickle.dump(latent_list, f)

    print('----- plot results ------')
    plot(latent_list, output_dir)

def plot(data, result_dir):
    color = 'navy'
    alpha = 0.5
    figure = plt.figure()
    ax = figure.add_subplot(111, aspect='equal', projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2], lw=0, c=color, alpha=alpha, label='trainings')
    ax.axis('tight')
    ax.legend(loc='best', shadow=False, scatterpoints=1)
    plt.savefig('{}/plot.png'.format(result_dir), dpi=120)
    with open('{}/plot.pickle'.format(result_dir), 'wb') as f:
        pickle.dump(figure, f)
    plt.show()

if __name__ == '__main__':
    main()
