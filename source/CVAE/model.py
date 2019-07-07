#coding:utf-8

"""
* @auther tozawa
* @date 2018-12-12
*
* Convolutional Variational Auto Encoder
"""

import os, sys, time
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
import chainer.distributions as D
from chainer import reporter

###############################################################################
# Utilty functions
###############################################################################
def get_valid_padding(kernel_size):
    """
    @param: kernel_size, kernel size of conv
    @return: spatial padding width
    'https://github.com/xinntao/ESRGAN/blob/master/block.py'
    """
    padding = (kernel_size - 1) // 2
    return padding

###############################################################################
# CVAE
###############################################################################
class Encoder(chainer.Chain):
    def __init__(self, in_channels=1, hidden_channels=16, out_channels=1, n_latent=3):
        self.n_latent = n_latent
        initializer = chainer.initializers.HeNormal()
        super(Encoder, self).__init__()
        with self.init_scope():
            self.ec0 = L.Convolution3D(in_channels=in_channels,
                                        out_channels=hidden_channels,
                                        ksize=4, stride=2, pad=1, initialW=initializer)
            self.bne0 = L.BatchNormalization(hidden_channels)
            self.ec1 = L.Convolution3D(in_channels=hidden_channels,
                                        out_channels=hidden_channels,
                                        ksize=3, stride=1, pad=1, initialW=initializer)
            self.bne1 = L.BatchNormalization(hidden_channels)

            self.ec2 = L.Convolution3D(in_channels=hidden_channels,
                                        out_channels=hidden_channels*2,
                                        ksize=4, stride=2, pad=1, initialW=initializer)
            self.bne2 = L.BatchNormalization(hidden_channels*2)
            self.ec3 = L.Convolution3D(in_channels=hidden_channels*2,
                                        out_channels=hidden_channels*2,
                                        ksize=3, stride=1, pad=1, initialW=initializer)
            self.bne3 = L.BatchNormalization(hidden_channels*2)

            self.ec4 = L.Convolution3D(in_channels=hidden_channels*2,
                                        out_channels=hidden_channels*4,
                                        ksize=4, stride=2, pad=1, initialW=initializer)
            self.bne4 = L.BatchNormalization(hidden_channels*4)
            self.ec5 = L.Convolution3D(in_channels=hidden_channels*4,
                                        out_channels=hidden_channels*4,
                                        ksize=3, stride=1, pad=1, initialW=initializer)
            self.bne5 = L.BatchNormalization(hidden_channels*4)

            self.ec6 = L.Convolution3D(in_channels=hidden_channels*4,
                                        out_channels=1,
                                        ksize=3, stride=1, pad=1, initialW=initializer)
            self.bne6 = L.BatchNormalization(1)

            self.mu = L.Linear(in_size=None, out_size=n_latent, initialW=initializer)
            self.ln_var = L.Linear(in_size=None, out_size=n_latent, initialW=initializer)

    def forward(self, x):
        h = F.relu(self.bne0(self.ec0(x)))
        h = F.relu(self.bne1(self.ec1(h)))
        h = F.relu(self.bne2(self.ec2(h)))
        h = F.relu(self.bne3(self.ec3(h)))
        h = F.relu(self.bne4(self.ec4(h)))
        h = F.relu(self.bne5(self.ec5(h)))
        h = F.relu(self.bne6(self.ec6(h)))
        mu = self.mu(h)
        ln_var = self.ln_var(h)
        return mu, ln_var

class Decoder(chainer.Chain):
    def __init__(self, in_channels=1, hidden_channels=16, out_channels=1, n_latent=3):
        initializer = chainer.initializers.HeNormal()
        self.n_latent = n_latent
        super(Decoder, self).__init__()
        with self.init_scope():
            self.hidden = L.Linear(in_size=None, out_size=(4*4*4), initialW=initializer)
            # S4
            self.dc6 = L.Deconvolution3D(in_channels=1,
                                        out_channels=hidden_channels*4,
                                        ksize=4, stride=2, initialW=initializer)
            self.dc5 = L.Convolution3D(in_channels=hidden_channels*4,
                                        out_channels=hidden_channels*4,
                                        ksize=3, stride=1, pad=0, initialW=initializer)
            self.bnd5 = L.BatchNormalization(hidden_channels*4)
            # S3
            self.dc4 = L.Deconvolution3D(in_channels=hidden_channels*4,
                                        out_channels=hidden_channels*2,
                                        ksize=4, stride=2, initialW=initializer)
            self.dc3 = L.Convolution3D(in_channels=hidden_channels*2,
                                        out_channels=hidden_channels*2,
                                        ksize=3, stride=1, pad=0, initialW=initializer)
            self.bnd3 = L.BatchNormalization(hidden_channels*2)
            # S2
            self.dc2 = L.Deconvolution3D(in_channels=hidden_channels*2,
                                        out_channels=hidden_channels,
                                        ksize=4, stride=2, initialW=initializer)
            self.dc1 = L.Convolution3D(in_channels=hidden_channels,
                                        out_channels=hidden_channels,
                                        ksize=3, stride=1, pad=0, initialW=initializer)
            self.bnd1 = L.BatchNormalization(hidden_channels)
            # S1
            self.dc0 = L.Deconvolution3D(in_channels=hidden_channels,
                                        out_channels=hidden_channels,
                                        ksize=4, stride=2, initialW=initializer)
            self.dc_1 = L.Convolution3D(in_channels=hidden_channels,
                                        out_channels=out_channels,
                                        ksize=3, stride=1, pad=0, initialW=initializer)

    def forward(self, x):
        h = F.relu(self.hidden(x))
        h = F.reshape(h, (h.shape[0], 1, 4, 4, 4))
        h = F.relu(self.bnd5(self.dc5(F.relu(self.dc6(h)))))
        h = F.relu(self.bnd3(self.dc3(F.relu(self.dc4(h)))))
        h = F.relu(self.bnd1(self.dc1(F.relu(self.dc2(h)))))
        h = F.relu(self.dc0(h))
        h = self.dc_1(h)
        return h

class AvgELBOLoss(chainer.Chain):
    """Loss function of CVAE.
    https://github.com/chainer/chainer/blob/master/examples/vae/net.py
    The loss value is equal to ELBO (Evidence Lower Bound)
    multiplied by -1.
    Args:
        encoder (chainer.Chain): A neural network which outputs variational
            posterior distribution q(z|x) of a latent variable z given
            an observed variable x.
        decoder (chainer.Chain): A neural network which outputs conditional
            distribution p(x|z) of the observed variable x given
            the latent variable z.
        prior (chainer.Chain): A prior distribution over the latent variable z.
        beta (float): Usually this is 1.0. Can be changed to control the
            second term of ELBO bound, which works as regularization.
        k (int): Number of Monte Carlo samples used in encoded vector.
    """
    def __init__(self, in_channels=1, hidden_channels=16, out_channels=1, n_latent=64,
                    beta=1.0, k=1):
        super(AvgELBOLoss, self).__init__()
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.beta = beta
        self.k = k
        with self.init_scope():
            self.encoder = Encoder(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, n_latent=n_latent)
            self.decoder = Decoder(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, n_latent=n_latent)

    def __call__(self, x):
        mu, ln_var = self.encoder(x)
        batchsize = len(mu.array)
        kl_penalty = F.gaussian_kl_divergence(mean=mu, ln_var=ln_var) / batchsize
        reconstr = 0
        for l in range(self.k):
            z = F.gaussian(mu, ln_var)
            recon = self.decoder(z)
            reconstr += 0.5 * F.mean_squared_error(recon, x) * x.shape[2] * x.shape[3] * x.shape[4] / self.k

        loss = (reconstr + self.beta * kl_penalty)
        reporter.report({'loss': loss}, self)
        reporter.report({'reconstr': reconstr}, self)
        reporter.report({'kl_penalty': kl_penalty}, self)
        return loss
