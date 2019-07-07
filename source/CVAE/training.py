#coding: utf-8
"""
* CVAE
"""
import os, sys, time, random
import numpy as np
import argparse, yaml, shutil
import chainer
from chainer import training
from chainer.training import extension
from chainer.training import extensions

from evaluators import generate_samples, projection_to_latent_space, reconstruction_img, CvaeEvaluator
from model import AvgELBOLoss
from dataset import CvaeDataset
sys.path.append(os.path.normpath(os.path.join(os.path.dirname(os.path.abspath( __file__ )), '../..')))
import util.yaml_utils  as yaml_utils

def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.backends.cuda.available:
        chainer.backends.cuda.cupy.random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description='Train CVAE')
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
    train = CvaeDataset(args.root,
                        os.path.join(args.base, config.dataset['training_fn']),
                        config.patch['patchside'],
                        [config.patch['lrmin'], config.patch['lrmax']],
                        augmentation=True)
    train_iter = chainer.iterators.SerialIterator(train, batch_size=config.batchsize)
    val = CvaeDataset(args.root,
                        os.path.join(args.base, config.dataset['val_fn']),
                        config.patch['patchside'],
                        [config.patch['lrmin'], config.patch['lrmax']],
                        augmentation=False)
    val_iter = chainer.iterators.SerialIterator(val, batch_size=config.batchsize, repeat=False, shuffle=False)

    print('----- Set up model ------')
    avg_elbo_loss = AvgELBOLoss()

    if args.gpu >= 0:
        chainer.backends.cuda.set_max_workspace_size(1024 * 1024 * 1024) # 1GB
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        avg_elbo_loss.to_gpu(args.gpu)

    print('----- Make optimizer -----')
    def make_optimizer(model, alpha=0.00001, beta1=0.9, beta2=0.999):
        optimizer = chainer.optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
        optimizer.setup(model)
        return optimizer
    gen_opt = make_optimizer(model = avg_elbo_loss,
                            alpha=config.adam['alpha'],
                            beta1=config.adam['beta1'],
                            beta2=config.adam['beta2'])

    print('----- Make updater -----')
    updater = training.updaters.StandardUpdater(
                iterator=train_iter,
                optimizer=gen_opt,
                device=args.gpu,
                loss_func=avg_elbo_loss)

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
            os.path.join(base_dir, config.dataset['fn']), result_dir)
        copy_to_result_dir(
            os.path.join(base_dir, config.dataset['training_fn']), result_dir)


    create_result_dir(args.base,  args.out, args.config_path, config)

    print('----- Make trainer -----')
    trainer = training.Trainer(updater,
                            (config.iteration, 'iteration'),
                            out=os.path.join(args.base, args.out))

    print('----- Set up logging -----')
    snapshot_interval = (config.snapshot_interval, 'iteration')
    display_interval = (config.display_interval, 'iteration')
    evaluation_interval = (config.evaluation_interval, 'iteration')
    trainer.extend(extensions.snapshot(filename='snapshot_iter_{.updater.iteration}.npz'),trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(avg_elbo_loss.encoder, filename='encoder_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    trainer.extend(extensions.snapshot_object(avg_elbo_loss.decoder, filename='decoder_iter_{.updater.iteration}.npz'), trigger=snapshot_interval)
    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(trigger=display_interval))
    # Evaluator
    trainer.extend(CvaeEvaluator(val_iter, avg_elbo_loss, device=args.gpu),
                   trigger=evaluation_interval)
    trainer.extend(generate_samples(avg_elbo_loss, os.path.join(args.base, args.out)),
                   trigger=evaluation_interval,
                   priority=extension.PRIORITY_WRITER)
    # trainer.extend(projection_to_latent_space(avg_elbo_loss, os.path.join(args.base, args.out), train),
    #                trigger=evaluation_interval,
    #                priority=extension.PRIORITY_WRITER)
    trainer.extend(reconstruction_img(avg_elbo_loss, os.path.join(args.base, args.out), train),
                   trigger=evaluation_interval,
                   priority=extension.PRIORITY_WRITER)
    # Print selected entries of the log to stdout
    #trainer.extend(extensions.PrintReport(['epoch', 'iteration', 'gen/loss', 'elapsed_time']), trigger=display_interval)
    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar(update_interval=10))
    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(extensions.PlotReport(['main/kl_penalty', 'validation/kl_penalty'], 'iteration', file_name='kl_loss.png', trigger=display_interval))
        trainer.extend(extensions.PlotReport(['main/reconstr', 'validation/reconstr'], 'iteration', file_name='reconstr_loss.png', trigger=display_interval))
        trainer.extend(extensions.PlotReport(['main/loss', 'validation/loss'], 'iteration', file_name='loss.png', trigger=display_interval))

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    reset_seed(0)
    trainer.run()

if __name__ == '__main__':
    main()
