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
from chainer.training import extensions
from chainer.training import extension
from evaluators import reconstruct_hr_img

sys.path.append(os.path.dirname(__file__))

from model import Generator_SR
from updater import preUpdater
from dataset import CycleganDataset
import util.yaml_utils  as yaml_utils


def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.backends.cuda.available:
        chainer.backends.cuda.cupy.random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description='Train pre')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                        help='base directory path of program files')
    parser.add_argument('--config_path', type=str, default='configs/pretraining.yml',
                        help='path to config file')
    parser.add_argument('--out', '-o', default= 'results/pretraining',
                        help='Directory to output the result')

    parser.add_argument('--model', '-m', default='',
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
    print('')

    print('----- Load dataset -----')
    train = CycleganDataset(args.root,
                            os.path.join(args.base, config.dataset['training_fn']),
                            config.patch['patchside'],
                            [config.patch['lrmin'], config.patch['lrmax']],
                            augmentation=False)
    train_iter = chainer.iterators.SerialIterator(train, batch_size=config.batchsize)

    print('----- Set up model ------')
    gen = Generator_SR()
    gen2 = Generator_SR()

    if args.model:
        chainer.serializers.load_npz(args.model, gen)
        chainer.serializers.load_npz(args.model2, gen2)

    if args.gpu >= 0:
        chainer.backends.cuda.set_max_workspace_size(1024 * 1024 * 1024) # 1GB
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
        gen2.to_gpu()


    print('----- Make optimizer -----')
    def make_optimizer(model, alpha=0.0001, beta1=0.9, beta2=0.999):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        optimizer.setup(model)
        return optimizer
    gen_opt = make_optimizer(model = gen,
                            alpha=config.adam['alpha'],
                            beta1=config.adam['beta1'],
                            beta2=config.adam['beta2'])

    gen2_opt = make_optimizer(model = gen2,
                            alpha=config.adam['alpha'],
                            beta1=config.adam['beta1'],
                            beta2=config.adam['beta2'])


    print('----- Make updater -----')
    updater = preUpdater(
        models = (gen,gen2),
        iterator = train_iter,
        optimizer = {'gen': gen_opt,'gen2':gen2_opt},
        device=args.gpu
        )

    print('----- Save configs -----')
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
    trainer.extend(extensions.snapshot_object(gen2, filename='gen2_iter_{.updater.iteration}.npz'),trigger=snapshot_interval)
    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(reconstruct_hr_img(gen,gen2, os.path.join(args.base, args.out), train), trigger=evaluation_interval,
                   priority=extension.PRIORITY_WRITER)


    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar(update_interval=10))

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(['loss_gen_pre'], 'iteration', file_name='srresnet_loss.png', trigger=display_interval))
        trainer.extend(
            extensions.PlotReport(['loss_cycLHL'], 'iteration', file_name='srresnet_loss_cyc_LHL.png', trigger=display_interval))
        trainer.extend(
            extensions.PlotReport(['loss_cycHLH'], 'iteration', file_name='srresnet_loss_cyc_HLH.png', trigger=display_interval))

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    #chainer.config.autotune = True
    print('----- Run the training -----')
    reset_seed(0)
    trainer.run()

if __name__ == '__main__':
    main()
