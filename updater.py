#coding:utf-8
"""
* Define updater
* @auther towawa
"""

#default
import os, time, sys
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
import SimpleITK as sitk

class SrganUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop("models")
        super(SrganUpdater, self).__init__(*args, **kwargs)
        self._lambda0 = 0.001

    def vanilla_dis_loss(self, y_fake, y_real):
        """
        vanilla loss e.g. dcgans loss
        Args:
            y_fake: Output when generator output is put into discriminator
            y_real: Output when real image is put into discriminator
        """
        L1 = F.mean(F.softplus(-y_real))
        L2 = F.mean(F.softplus(y_fake))
        loss = L1 + L2
        #chainer.reporter.report({'dis/L1':L1, 'dis/L2':L2})
        return loss

    def vanilla_gen_loss(self, y_fake):
        """
        vanilla loss for generator
        Args:
            y_fake: Output when generator output is put into discriminator
        """
        loss = F.mean(F.softplus(-y_fake))
        return loss

    def hinge_dis_loss(self, y_fake, y_real):
        L1 = F.mean(F.relu(1. - y_real))
        L2 = F.mean(F.relu(1. + y_fake))
        loss = L1 + L2
        return loss

    def hinge_gen_loss(self, y_fake):
        loss = -F.mean(y_fake)
        return loss

    def mse_loss(self, x_real, x_fake):
        """
        MSE loss
        Args:
            x_real: ground truth
            x_fake: generator output
        """
        loss = F.mean_squared_error(x_real, x_fake)
        return loss

    def rotation_loss(self, predict, ground_truth):
        """
        Auxiliary rotation loss
        Args:
            predict: predicted label
            ground_truth:
        """
        loss = F.softmax_cross_entropy(predict, ground_truth)
        return loss

    def weighted_cross_entropy(self, gt, logits, xp, dims=(2,3,4)):
        batchsize, ch, D, H, W = logits.shape
        pos_weight = F.reshape(F.sum(F.softmax(logits), axis=dims)[:, 1], (-1, 1))
        pos_weight = (D*H*W - pos_weight) / pos_weight
        weight = F.concat([chainer.Variable(xp.ones((batchsize, 1), dtype=xp.float32)), pos_weight], axis=1)

        loss = -1 * F.mean(F.sum(F.log_softmax(logits) * gt, axis=dims) * weight)
        return loss

    def update_core(self):
        gen_optimizer = self.get_optimizer("gen")#load optimizer called "gen"
        dis_optimizer = self.get_optimizer("dis")

        batch = self.get_iterator("main").next()#iterator
        lr_real, hr_real = self.converter(batch, self.device)

        gen, dis = self.gen, self.dis#generator and discriminator

        """
        Update for discriminator
        """
        y_real = dis(hr_real)

        hr_fake = gen(lr_real)

        y_fake = dis(hr_fake)

        dis_loss = self.vanilla_dis_loss(y_fake=y_fake, y_real=y_real)
        dis.cleargrads()
        dis_loss.backward()
        dis_optimizer.update()
        chainer.reporter.report({'dis_loss':dis_loss})

        """
        Update for generator
        """
        gen_mse_loss = self.mse_loss(x_real=hr_real, x_fake=hr_fake)
        gen_adv_loss = self.vanilla_gen_loss(y_fake=y_fake)

        gen_loss = gen_mse_loss + self._lambda0 * gen_adv_loss
        gen.cleargrads()
        gen_loss.backward()
        gen_optimizer.update()
        chainer.reporter.report({'gen_loss':gen_loss,
                                'gen_adv_loss':gen_adv_loss,
                                'gen_mse_loss':gen_mse_loss})
