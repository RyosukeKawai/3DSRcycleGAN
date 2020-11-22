#coding: utf-8
import os, sys, time, random
import numpy as np
import argparse, yaml, shutil
import chainer
from chainer import training
from chainer.datasets import TransformDataset
from chainer.training import extension
from chainer.training import extensions

sys.path.append(os.path.dirname(__file__))

from interpolate_resnet import Generator
from encoder_based_dis import Discriminator

from updater import SrganUpdater
from dataset import InterpolateDataset
from evaluators import reconstruct_hr_img, segment_hr_img
import utils.yaml_utils  as yaml_utils
import utils.ioFunctions as IO
from utils.transform import transform, flip_transform

def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.backends.cuda.available:
        chainer.backends.cuda.cupy.random.seed(seed)

def save_file(args, configs):
    """https://github.com/pfnet-research/sngan_projection/blob/master/train.py"""
    base_dir = args.base
    result_dir = os.path.join(base_dir, args.out)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists('{}/init'.format(result_dir)):
        os.makedirs('{}/init'.format(result_dir))

    def copy_to_result_dir(fn, result_dir):
        bfn = os.path.basename(fn)
        shutil.copy(fn, '{}/{}'.format(result_dir, bfn))

    copy_to_result_dir(
        os.path.join(base_dir, configs.filename['training']), result_dir)
    copy_to_result_dir(
        os.path.join(base_dir, configs.filename['updater']), result_dir)
    # copy_to_result_dir(
    #     os.path.join(base_dir, configs.filename['gen_model']), result_dir)
    # copy_to_result_dir(
    #     os.path.join(base_dir, configs.filename['dis_model']), result_dir)
    copy_to_result_dir(
        os.path.join(base_dir, configs.filename['dataset']), result_dir)
    # copy_to_result_dir(
    #     os.path.join(base_dir, configs.filename['evaluator']), result_dir)
    # copy_to_result_dir(
    #     os.path.join(base_dir, configs.dataset['training_list']), result_dir)
    # copy_to_result_dir(
    #     os.path.join(base_dir, configs.model['gen_init']), '{}/init'.format(result_dir))
    copy_to_result_dir(
        os.path.join(base_dir, args.config_path), result_dir)
    IO.save_args(result_dir, args)
    return

def main():
    parser = argparse.ArgumentParser(description='Train SRGAN')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--base', '-B', default=os.path.dirname(os.path.abspath(__file__)),
                        help='base directory path of program files')
    parser.add_argument('--config_path', type=str, default='configs/training.yml',
                        help='path to config file')
    parser.add_argument('--out', '-o', default= 'results/training_{}'.format(time.strftime('%Y-%m-%d_%H-%M-%S')),
                        help='Directory to output the result')

    parser.add_argument('--model', '-m', default='',
                        help='Load model data')
    parser.add_argument('--resume', '-res', default='',
                        help='Resume the training from snapshot')

    parser.add_argument('--root', '-R', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Root directory path of input data')

    parser.add_argument('--train', '-T', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Train data type')

    args = parser.parse_args()

    config = yaml_utils.Config(yaml.load(open(os.path.join(args.base, args.config_path))))

    print('----- Save configs -----')
    save_file(args, config)

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(config.batchsize))
    print('# iteration: {}'.format(config.iteration))
    print('Learning Rate: {}'.format(config.adam['alpha']))
    print('')


    # Read path to hr data and lr data
    path_pairs = []
    with open(os.path.join(args.base, config.dataset[args.train])) as paths_file:
        for line in paths_file:
            line = line.split()
            if not line: continue
            path_pairs.append(line[:])

    min =float(path_pairs[2][0])
    max =float(path_pairs[2][1])

    print('----- Set up model ------')
    gen = Generator()
    if args.model:
        print('Initialize generator parameters')
        chainer.serializers.load_npz(
                                    args.model, gen)
    dis = Discriminator()
    if args.gpu >= 0:
        chainer.global_config.autotune = True
        chainer.backends.cuda.set_max_workspace_size(512 * 1024 * 1024) # 1GB
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        gen.to_gpu()
        dis.to_gpu()

    print('----- Load dataset -----')
    train = InterpolateDataset(args.root,
                        os.path.join(args.base, config.dataset[args.train]),
                        config.patch['patchside'],
                        [min, max])
    transformed_train = TransformDataset(train, flip_transform)
    train_iter = chainer.iterators.MultiprocessIterator(transformed_train, batch_size=config.batchsize, n_processes=config.batchsize)



    print('----- Make optimizer -----')
    def make_optimizer(model, alpha=0.00001, beta1=0.9, beta2=0.999):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        optimizer.setup(model)
        return optimizer
    gen_opt = make_optimizer(model = gen,
                            alpha=config.adam['alpha'],
                            beta1=config.adam['beta1'],
                            beta2=config.adam['beta2'])
    dis_opt = make_optimizer(model = dis,
                            alpha=config.adam['alpha'],
                            beta1=config.adam['beta1'],
                            beta2=config.adam['beta2'])

    print('----- Make updater -----')
    updater = SrganUpdater(
        models = (gen, dis),
        iterator = train_iter,
        optimizer = {'gen': gen_opt, 'dis': dis_opt},
        device=args.gpu
        )

    print('----- Make trainer -----')
    trainer = training.Trainer(updater,
                            (config.iteration, 'iteration'),
                            out=os.path.join(args.base, args.out))

    # Set up logging
    snapshot_interval = (config.snapshot_interval, 'iteration')
    display_interval = (config.display_interval, 'iteration')
    evaluation_interval = (config.evaluation_interval, 'iteration')
    trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(gen, filename='gen_iter_{.updater.iteration}.npz'), trigger=evaluation_interval)
    trainer.extend(extensions.snapshot_object(dis, filename='dis_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.LogReport(trigger=display_interval))
    trainer.extend(reconstruct_hr_img(gen, os.path.join(args.base, args.out), train),
                   trigger=evaluation_interval,
                   priority=extension.PRIORITY_WRITER)
    # Print selected entries of the log to stdout
    #trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'gen/loss', 'elapsed_time']), trigger=display_interval)
    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar(update_interval=10))
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(['gen_loss'], 'iteration', file_name='gen_loss.png', trigger=display_interval))
        trainer.extend(extensions.PlotReport(['gen_mse'], 'iteration', file_name='gen_mse_loss.png', trigger=display_interval))
        trainer.extend(extensions.PlotReport(['gen_adv_loss'], 'iteration', file_name='gen_adv_loss.png', trigger=display_interval))
        trainer.extend(extensions.PlotReport(['dis_loss'], 'iteration', file_name='dis_loss.png', trigger=display_interval))

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    print('----- Run the training -----')
    reset_seed(0)
    trainer.run()

if __name__ == '__main__':
    main()
