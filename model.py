#coding:utf-8


"""
* @auther ryosuke
* reference source :https://github.com/zEttOn86/3D-SRGAN
"""

#default
import sys, os, time, math
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
import chainer.link_hooks
from util.links.sn_convolution_nd import SNConvolutionND
from util.links.sn_linear import SNLinear
from util.layers.difference_filter import EdgeEnhanceFilter
from util.layers.self_attention import SelfAttention3D


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

def cropping(input,ref):
    ref_map=np.zeros((ref,ref,ref))
    rZ, rY, rX =ref_map.shape
    _, _, iZ, iY, iX = input.shape
    edgeZ, edgeY, edgeX = (iZ - rZ)//2, (iY - rY)//2, (iX - rX)//2
    edgeZZ, edgeYY, edgeXX = iZ-edgeZ, iY-edgeY, iX-edgeX

    _, X, _ = F.split_axis(input, (edgeX, edgeXX), axis=4)
    _, X, _ = F.split_axis(X, (edgeY, edgeYY), axis=3)
    _, X, _ = F.split_axis(X, (edgeZ, edgeZZ), axis=2)

    return X

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
                 out_channels=32,
                 num_resblocks=8):
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

###############################################################################
# Generator2
###############################################################################

class Generator2(chainer.Chain):
        def __init__(self,
                     in_channels=1,
                     out_channels=32,
                     num_resblocks=8):
            super(Generator2, self).__init__()
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
                                                        out_channels=out_channels,stride=2,
                                                        ksize=3, pad=1, initialW=initializer)
                self.upsampling_conv2 = L.Convolution3D(in_channels=out_channels,
                                                        out_channels=out_channels,stride=2,
                                                        ksize=3, pad=1, initialW=initializer)
                self.upsampling_conv3 = L.Convolution3D(in_channels=out_channels,
                                                        out_channels=out_channels,stride=2,
                                                        ksize=3, pad=1, initialW=initializer)

                self.last_conv = L.Convolution3D(in_channels=out_channels,
                                                 out_channels=1,
                                                 ksize=5, pad=get_valid_padding(5), initialW=initializer)

        def to_cpu(self):
            super(Generator2, self).to_cpu()

        def to_gpu(self, device=None):
            super(Generator2, self).to_gpu(device)

        def forward(self, x):
            h1 = F.leaky_relu(self.conv1(x))
            h = self.bn2(self.conv2(self.resblock(h1)))
            h = h + h1  # Skip connection
            h = F.leaky_relu(self.upsampling_conv1(h))
            h = F.leaky_relu(self.upsampling_conv2(h))
            h = F.leaky_relu(self.upsampling_conv3(h))
            h = self.last_conv(h)
            return h


###############################################################################
# Discriminator
###############################################################################
class Discriminator(chainer.Chain):

    def __init__(self, ch=32):
        w = chainer.initializers.Normal(scale=0.02)#Inspired by DCGAN
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.conv1 = L.ConvolutionND(ndim=3, in_channels=1, out_channels=ch, ksize=3, stride=1, pad=1, initialW=w).add_hook(chainer.link_hooks.SpectralNormalization())

            self.conv2 = L.ConvolutionND(ndim=3, in_channels=ch, out_channels=ch, ksize=3, stride=2, pad=1, initialW=w).add_hook(chainer.link_hooks.SpectralNormalization())


            self.conv3 = L.ConvolutionND(ndim=3, in_channels=ch, out_channels=ch*2, ksize=3, stride=1, pad=1, initialW=w).add_hook(chainer.link_hooks.SpectralNormalization())


            self.conv4 = L.ConvolutionND(ndim=3, in_channels=ch*2, out_channels=ch*2, ksize=3, stride=2, pad=1, initialW=w).add_hook(chainer.link_hooks.SpectralNormalization())


            self.conv5 = L.ConvolutionND(ndim=3, in_channels=ch*2, out_channels=ch*4, ksize=3, stride=1, pad=1, initialW=w).add_hook(chainer.link_hooks.SpectralNormalization())


            self.conv6 = L.ConvolutionND(ndim=3, in_channels=ch *4, out_channels=ch*4, ksize=3, stride=2, pad=1,initialW=w).add_hook(chainer.link_hooks.SpectralNormalization())


            self.conv7 = L.ConvolutionND(ndim=3, in_channels=ch*4, out_channels=ch*8, ksize=3, stride=1, pad=1, initialW=w).add_hook(chainer.link_hooks.SpectralNormalization())


            self.conv8 = L.ConvolutionND(ndim=3, in_channels=ch*8, out_channels=ch*8,
                                         ksize=3, stride=2, pad=1, initialW=w).add_hook(chainer.link_hooks.SpectralNormalization())


            self.fc1 = L.Linear(None, ch*8, initialW=w)
            self.fc2 = L.Linear(ch*8, 1, initialW=w)

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x))
        h = F.leaky_relu(self.conv2(h))
        h = F.leaky_relu(self.conv3(h))
        h = F.leaky_relu(self.conv4(h))
        h = F.leaky_relu(self.conv5(h))
        # h = F.leaky_relu(self.conv7(h))
        # h = F.leaky_relu(self.conv8(h))
        h = F.leaky_relu(self.fc1(h))
        h = self.fc2(h)
        return h


###############################################################################
# Discriminator
###############################################################################
class Discriminator2(chainer.Chain):

    def __init__(self, ch=32):
        w = chainer.initializers.Normal(scale=0.02)#Inspired by DCGAN
        super(Discriminator2, self).__init__()
        with self.init_scope():
            self.conv1 = L.ConvolutionND(ndim=3, in_channels=1, out_channels=ch, ksize=3, stride=1, pad=1, initialW=w).add_hook(chainer.link_hooks.SpectralNormalization())

            self.conv2 = L.ConvolutionND(ndim=3, in_channels=ch, out_channels=ch, ksize=3, stride=2, pad=1, initialW=w).add_hook(chainer.link_hooks.SpectralNormalization())


            self.conv3 = L.ConvolutionND(ndim=3, in_channels=ch, out_channels=ch*2, ksize=3, stride=1, pad=1, initialW=w).add_hook(chainer.link_hooks.SpectralNormalization())


            self.conv4 = L.ConvolutionND(ndim=3, in_channels=ch*2, out_channels=ch*2, ksize=3, stride=2, pad=1, initialW=w).add_hook(chainer.link_hooks.SpectralNormalization())


            self.conv5 = L.ConvolutionND(ndim=3, in_channels=ch*2, out_channels=ch*4, ksize=3, stride=1, pad=1, initialW=w).add_hook(chainer.link_hooks.SpectralNormalization())


            self.conv6 = L.ConvolutionND(ndim=3, in_channels=ch *4, out_channels=ch*4, ksize=3, stride=2, pad=1,initialW=w).add_hook(chainer.link_hooks.SpectralNormalization())


            self.conv7 = L.ConvolutionND(ndim=3, in_channels=ch*4, out_channels=ch*8, ksize=3, stride=1, pad=1, initialW=w).add_hook(chainer.link_hooks.SpectralNormalization())


            self.conv8 = L.ConvolutionND(ndim=3, in_channels=ch*8, out_channels=ch*8,
                                         ksize=3, stride=2, pad=1, initialW=w).add_hook(chainer.link_hooks.SpectralNormalization())


            self.fc1 = L.Linear(None, ch*8, initialW=w)
            self.fc2 = L.Linear(ch*8, 1, initialW=w)

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x))
        h = F.leaky_relu(self.conv2(h))
        h = F.leaky_relu(self.conv3(h))
        h = F.leaky_relu(self.conv4(h))
        h = F.leaky_relu(self.conv5(h))
        # h = F.leaky_relu(self.conv7(h))
        # h = F.leaky_relu(self.conv8(h))
        h = F.leaky_relu(self.fc1(h))
        h = self.fc2(h)
        return h