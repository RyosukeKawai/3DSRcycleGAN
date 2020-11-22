#coding:utf-8
import sys, os, time, math
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
import chainer.link_hooks

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
# Generator
###############################################################################
class ResidualBlock(chainer.Chain):
    """
    * single residual block
    """
    def __init__(self, in_channels=64, hidden_channels=None, out_channels=64, ksize=3):
        super(ResidualBlock, self).__init__()
        initializer = chainer.initializers.HeNormal()
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        with self.init_scope():
            self.conv1 = L.Convolution3D(in_channels=in_channels, out_channels=hidden_channels, ksize=ksize, pad=get_valid_padding(ksize), initialW=initializer)
            self.bn1 = L.BatchNormalization(hidden_channels)
            self.conv2 = L.Convolution3D(in_channels=hidden_channels, out_channels=out_channels, ksize=ksize, pad=get_valid_padding(ksize), initialW=initializer)
            self.bn2 = L.BatchNormalization(out_channels)

    def forward(self, x):
        h1 = F.leaky_relu(self.bn1(self.conv1(x)))
        h1 = self.bn2(self.conv2(h1))
        return (h1 + x)

class Generator(chainer.Chain):
    def __init__(self,
                 in_channels=1,
                 out_channels=64,
                 num_resblocks=16):
        super(Generator, self).__init__()
        initializer = chainer.initializers.HeNormal()
        with self.init_scope():
            self.conv1 = L.Convolution3D(in_channels=in_channels,
                                         out_channels=out_channels,
                                         ksize=5, pad=get_valid_padding(5), initialW=initializer)

            self.resblock = ResidualBlock(in_channels=out_channels,
                                          out_channels=out_channels,
                                          ksize=3).repeat(num_resblocks, mode='init')
            self.conv2 = L.Convolution3D(in_channels=out_channels,
                                         out_channels=out_channels,
                                         ksize=3, pad=get_valid_padding(3), initialW=initializer)
            self.bn2 = L.BatchNormalization(out_channels)

            self.upsampling_conv1 = L.Convolution3D(in_channels=out_channels,
                                                     out_channels=out_channels,
                                                     ksize=3, pad=1, initialW=initializer)
            self.upsampling_conv2 = L.Convolution3D(in_channels=out_channels,
                                             out_channels=out_channels,
                                             ksize=3, pad=1, initialW=initializer)
            self.upsampling_conv3 = L.Convolution3D(in_channels=out_channels,
                                             out_channels=out_channels,
                                             ksize=3, pad=1, initialW=initializer)

            self.last_conv = L.Convolution3D(in_channels=out_channels,
                                             out_channels=1,
                                             ksize=5, pad=get_valid_padding(5), initialW=initializer)
    def to_cpu(self):
        super(Generator, self).to_cpu()

    def to_gpu(self, device=None):
        super(Generator, self).to_gpu(device)

    def forward(self, x):
        h1 = F.leaky_relu(self.conv1(x))
        h = self.bn2(self.conv2(self.resblock(h1)))
        h = h + h1 # Skip connection
        h = F.unpooling_3d(h, ksize=2, outsize=(16,16,16))
        h = F.leaky_relu(self.upsampling_conv1(h))
        h = F.unpooling_3d(h, ksize=2, outsize=(32,32,32))
        h = F.leaky_relu(self.upsampling_conv2(h))
        h = F.unpooling_3d(h, ksize=2, outsize=(64,64,64))
        h = F.leaky_relu(self.upsampling_conv3(h))
        h = self.last_conv(h)
        return h

class MeanSquaredErrorLoss(chainer.Chain):
    def __init__(self,
                 in_channels=1,
                 out_channels=64,
                 num_resblocks=16):
        super(MeanSquaredErrorLoss, self).__init__()
        with self.init_scope():
            self.model = Generator(in_channels=1,
                                    out_channels=64,
                                    num_resblocks=16)

    def __call__(self, lr, hr):
        hr_hat = self.model(lr)
        loss = F.mean_squared_error(hr, hr_hat)
        chainer.report({"gen/loss": loss})

        return loss
