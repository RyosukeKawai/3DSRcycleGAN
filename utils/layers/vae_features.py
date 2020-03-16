# coding:utf-8
import os, sys, time
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

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

    def forwardV2(self, x):
        h = F.relu(self.bne0(self.ec0(x)))
        h1 = F.relu(self.bne1(self.ec1(h)))
        h = F.relu(self.bne2(self.ec2(h1)))
        h2 = F.relu(self.bne3(self.ec3(h)))
        h = F.relu(self.bne4(self.ec4(h2)))
        h3 = F.relu(self.bne5(self.ec5(h)))
        h = F.relu(self.bne6(self.ec6(h3)))
        mu = self.mu(h)
        #ln_var = self.ln_var(h)
        return h1, h2, h3, mu

class VaeFeatures(chainer.Chain):
    def __init__(self, in_channels=1, hidden_channels=16, out_channels=1, n_latent=64,
                    model_pass = 'source/CVAE/results/training0012/encoder_iter_100000.npz'):
        """
        This loss function compare features in vae latent space.
        """
        super(VaeFeatures, self).__init__()
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        with self.init_scope():
            self.encoder = Encoder(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, n_latent=n_latent)

        print('----- Initialize encoder -----')
        chainer.serializers.load_npz(model_pass, self.encoder)

    def __call__(self, x, gt):
        enc_gen, _ = self.encoder(x)
        enc_gt, _  = self.encoder(gt)

        loss = F.mean_squared_error(enc_gen, enc_gt)

        return loss

class VaeFeaturesV2(chainer.Chain):
    def __init__(self, in_channels=1, hidden_channels=16, out_channels=1, n_latent=64,
                    model_pass = 'source/CVAE/results/training0012/encoder_iter_100000.npz',
                    lambda0=1., lambda1=0.5, lambda2=1., lambda3=10, lambda4=10):
        """
        This loss function compare features in vae latent space.
        """
        super(VaeFeaturesV2, self).__init__()
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.lambda0 = lambda0
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.lambda4 = lambda4
        with self.init_scope():
            self.encoder = Encoder(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, n_latent=n_latent)

        print('----- Initialize encoder -----')
        chainer.serializers.load_npz(model_pass, self.encoder)

    def __call__(self, x, gt):
        enc_gen1, enc_gen2, enc_gen3, enc_gen4 = self.encoder.forwardV2(x)
        enc_gt1, enc_gt2, enc_gt3, enc_gt4   = self.encoder.forwardV2(gt)
        L0 = F.mean_squared_error(x, gt)
        L1 = F.mean_squared_error(enc_gen1, enc_gt1)
        L2 = F.mean_squared_error(enc_gen2, enc_gt2)
        L3 = F.mean_squared_error(enc_gen3, enc_gt3)
        L4 = F.mean_squared_error(enc_gen4, enc_gt4)
        loss = self.lambda0*L0 + self.lambda1*L1 + self.lambda2*L2 + self.lambda3*L3 + self.lambda4*L4

        chainer.reporter.report({'enc/L0':L0, 'enc/L1':L1, 'enc/L2':L2, 'enc/L3':L3, 'enc/L4':L4})

        return loss

class VaeFeaturesV3(chainer.Chain):
    def __init__(self, in_channels=1, hidden_channels=16, out_channels=1, n_latent=64,
                    model_pass = 'source/CVAE/results/training0012/encoder_iter_100000.npz',
                    lambda1=1.):
        """
        This loss function compare features in vae latent space.
        """
        super(VaeFeaturesV3, self).__init__()
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.lambda1 = lambda1
        with self.init_scope():
            self.encoder = Encoder(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=out_channels, n_latent=n_latent)

        print('----- Initialize encoder -----')
        chainer.serializers.load_npz(model_pass, self.encoder)

    def __call__(self, x, gt):
        enc_gen1, _, _, _ = self.encoder.forwardV2(x)
        enc_gt1, _, _, _   = self.encoder.forwardV2(gt)
        L1 = F.mean_squared_error(enc_gen1, enc_gt1)
        loss = self.lambda1*L1

        chainer.reporter.report({'enc/L1':L1})

        return loss
