#coding: utf-8
"""
* @auther ryosuke
* reference source :https://github.com/zEttOn86/3D-SRGAN
"""
import os, sys, time, random
import numpy as np
import argparse, yaml, shutil
import chainer
from chainer import training
from chainer.datasets import TransformDataset

sys.path.append(os.path.dirname(__file__))

from model import Generator, Discriminator,Generator2
from updater import CinCGANUpdater
from dataset import CycleganDataset
import utils.yaml_utils  as yaml_utils
from utils.transform import transform
from evaluators import reconstruct_hr_img
from chainer.training import extension
from chainer.training import extensions

def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.backends.cuda.available:
        chainer.backends.cuda.cupy.random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description='Train CycleGAN')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                        help='base directory path of program files')
    parser.add_argument('--config_path', type=str, default='configs/training.yml',
                        help='path to config file')
    parser.add_argument('--out', '-o', default= 'results/training',
                        help='Directory to output the result')

    parser.add_argument('--model', '-m', default='',
                        help='Load model data')

    parser.add_argument('--model2', '-m2', default='',
                        help='Load model data')


    parser.add_argument('--resume', '-res', default='',
                        help='Resume the training from snapshot')

    parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Root directory path of input image')

    args = parser.parse_args()

    config = yaml_utils.Config(yaml.load(open(os.path.join(args.base, args.config_path))))

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(config.batchsize))
    print('# iteration: {}'.format(config.iteration))
    print('Learning Rate: {}'.format(config.adam['alpha']))
    print('')

    #load the dataset
    print('----- Load dataset -----')
    train = CycleganDataset(args.root,
                            os.path.join(args.base, config.dataset['training_fn']),
                            config.patch['patchside'],
                            [config.patch['lrmin'], config.patch['lrmax']],)
    transformed_train=TransformDataset(train,transform)
    train_iter = chainer.iterators.MultiprocessIterator(transformed_train, batch_size=config.batchsize,n_processes=config.batchsize)

    print('----- Set up model ------')
    gen = Generator()
    #gen2 = Generator2()
    disY = Discriminator()
    #disX = Discriminator()
    # chainer.serializers.load_npz(args.model, gen)
    # chainer.serializers.load_npz(args.model2, gen2)


    if args.gpu >= 0:
        chainer.backends.cuda.set_max_workspace_size(1024 * 1024 * 1024) # 1GB
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
        #gen2.to_gpu()
        disY.to_gpu()
        #disX.to_gpu()


    print('----- Make optimizer -----')
    def make_optimizer(model, alpha=0.0001, beta1=0.9, beta2=0.999):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        optimizer.setup(model)
        return optimizer
    gen_opt = make_optimizer(model = gen,
                            alpha=config.adam['alpha'],
                            beta1=config.adam['beta1'],
                            beta2=config.adam['beta2'])

    # gen2_opt = make_optimizer(model = gen2,
    #                         alpha=config.adam['alpha'],
    #                         beta1=config.adam['beta1'],
    #                         beta2=config.adam['beta2'])

    disY_opt = make_optimizer(model = disY,
                            alpha=config.adam['alpha'],
                            beta1=config.adam['beta1'],
                            beta2=config.adam['beta2'])

    # disX_opt = make_optimizer(model = disX,
    #                         alpha=config.adam['alpha'],
    #                         beta1=config.adam['beta1'],
    #                         beta2=config.adam['beta2'])

    print('----- Make updater -----')
    updater = CinCGANUpdater(
        models = (gen,disY),
        iterator = train_iter,
        optimizer={'gen': gen_opt,'disY': disY_opt},
        device=args.gpu
        )

    print('----- Save configs -----')
    def create_result_dir(base_dir, output_dir, config_path, config):
        """https://github.com/pfnet-research/sngan_projection/blob/master/train.py"""
        result_dir = os.path.join(base_dir, output_dir)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        if not os.path.exists('{}/init'.format(result_dir)):
            os.makedirs('{}/init'.format(result_dir))

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
            os.path.join(base_dir, config.dataset['training_fn']), result_dir)

    create_result_dir(args.base,  args.out, args.config_path, config)

    print('----- Make trainer -----')
    trainer = training.Trainer(updater,
                            (config.iteration, 'iteration'),
                            out=os.path.join(args.base, args.out))

    # Set up logging
    snapshot_interval = (config.snapshot_interval, 'iteration')
    display_interval = (config.display_interval, 'iteration')
    evaluation_interval = (config.evaluation_interval, 'iteration')
    trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(gen, filename='gen_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    #trainer.extend(extensions.snapshot_object(gen2, filename='gen2_iter_{.updater.iteration}.npz'),trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(disY, filename='disY_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    #trainer.extend(extensions.snapshot_object(disX, filename='disX_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(reconstruct_hr_img(gen,os.path.join(args.base, args.out),train),trigger=evaluation_interval,priority=extension.PRIORITY_WRITER)

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(['gen/loss_gen1'], 'iteration', file_name='gen_loss.png', trigger=display_interval))
        trainer.extend(extensions.PlotReport(['disY/loss_dis1_fake', 'disY/loss_dis1_real',
                                              'disY/loss_dis1'], 'iteration', file_name='dis_loss.png', trigger=display_interval))
        trainer.extend(extensions.PlotReport(['gen/loss_gen',
                                               'disY/loss_dis1'], 'iteration', file_name='adv_loss.png',trigger=display_interval))

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    print('----- Run the training -----')
    reset_seed(0)
    trainer.run()

if __name__ == '__main__':
    main()


