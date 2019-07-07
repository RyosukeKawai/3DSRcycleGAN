#coding:utf-8

"""
* define generator and discriminator of srgan
* @auther towawa
* @history
* 20180629 Refactoring
* 20180105 input is interpolate patch
* 20171218 Add 'F.sigmoid' to Discriminator output
* 20171208 Add ShuffleBlock for upsamplingRate 8
*          Assmue Patch side is 32.
* 20171031 All weights were initialized from a zero-centered Normal distribution with standard deviation 0.02
* 20171024 made
"""

#default
import sys, os, time
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F

from util.links.sn_convolution_nd import SNConvolutionND
from util.links.sn_linear import SNLinear

def get_valid_padding(kernel_size):
    """
    @param: kernel_size, kernel size of conv
    @return: spatial padding width
    'https://github.com/xinntao/ESRGAN/blob/master/block.py'
    """
    padding = (kernel_size - 1) // 2
    return padding

def pad(pad_type, padding):
    """
    @param: pad_type, type of padding
    @param: padding, spatial padding width for one side
    """
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    else:
        raise NotImplementedError('padding layer [{}] is not implemented'.format(pad_type))

class ResidualDenseBlock(chainer.Chain):
    """
    * Residual Dense Block
    * 'https://github.com/xinntao/ESRGAN/blob/master/block.py'
    """
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad_type='zero'):
        w = chainer.initializers.HeNormal(scale=0.9701425)
        super(ResidualDenseBlock, self).__init__()
        padding = get_valid_padding(ksize)
        #p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
        #padding = padding if pad_type == 'zero' else 0

        with self.init_scope():
            self.conv1 = L.ConvolutionND(ndim=3, in_channels=in_channels, out_channels=out_channels, ksize=ksize, pad=padding, initialW=w)
            self.conv2 = L.ConvolutionND(ndim=3, in_channels=in_channels+out_channels, out_channels=out_channels, ksize=ksize, pad=padding, initialW=w)
            self.conv3 = L.ConvolutionND(ndim=3, in_channels=in_channels+2*out_channels, out_channels=out_channels, ksize=ksize, pad=padding, initialW=w)
            self.conv4 = L.ConvolutionND(ndim=3, in_channels=in_channels+3*out_channels, out_channels=out_channels, ksize=ksize, pad=padding, initialW=w)
            self.conv5 = L.ConvolutionND(ndim=3, in_channels=in_channels+4*out_channels, out_channels=in_channels, ksize=ksize, pad=padding, initialW=w)

    def forward(self, x):
        x1 = F.leaky_relu(self.conv1(x))
        x2 = F.leaky_relu(self.conv2(F.concat([x, x1])))
        x3 = F.leaky_relu(self.conv3(F.concat([x, x1, x2])))
        x4 = F.leaky_relu(self.conv4(F.concat([x, x1, x2, x3])))
        x5 = F.leaky_relu(self.conv5(F.concat([x, x1, x2, x3, x4])))
        return x5 + x

class RRDB(chainer.Chain):
    """
    * Residual in Residual Dense Block
    * 'https://github.com/xinntao/ESRGAN/blob/master/block.py'
    """
    def __init__(self, in_channels, out_channels, ksize=3, stride=1, pad_type='zero'):
        w = chainer.initializers.HeNormal(scale=0.9701425)
        super(RRDB, self).__init__()
        padding = get_valid_padding(ksize)

        with self.init_scope():
            self.RDB1 = ResidualDenseBlock(in_channels, out_channels, ksize=ksize, stride=stride, pad_type=pad_type)
            self.RDB2 = ResidualDenseBlock(out_channels, out_channels, ksize=ksize, stride=stride, pad_type=pad_type)
            self.RDB3 = ResidualDenseBlock(out_channels, in_channels, ksize=ksize, stride=stride, pad_type=pad_type)

    def forward(self, x):
        h = self.RDB1(x)
        h = self.RDB2(h)
        h = self.RDB3(h)
        return h + x

class RRDBList(chainer.ChainList):
    """
    * Residual in Residual Dense Block List
    * @param layer # of residual block
    """
    def __init__(self, layer, in_channels, out_channels, ksize=3, stride=1, pad_type='zero'):
        super(RRDBList, self).__init__()
        for i in range(layer):
            self.add_link(RRDB(in_channels, out_channels, ksize=3, stride=1, pad_type=pad_type))

    def forward(self, x):
        for f in self.children():
            x = f(x)
        return x

class SkipConnection(chainer.Chain):
    def __init__(self, layer, in_channels, out_channels, ksize=3, stride=1, pad_type='zero'):
        w = chainer.initializers.HeNormal(scale=0.9701425)
        super(SkipConnection, self).__init__()

        with self.init_scope():
            self.rrdblock = RRDBList(layer=layer, in_channels=in_channels, out_channels=out_channels, ksize=ksize, stride=stride, pad_type=pad_type)
            self.conv1 = L.ConvolutionND(ndim=3, in_channels=out_channels, out_channels=in_channels, ksize=ksize, pad=get_valid_padding(ksize), initialW=w)

    def forward(self, x):
        return self.conv1(self.rrdblock(x)) + x

class Generator(chainer.Chain):

    def __init__(self):
        w = chainer.initializers.HeNormal(scale=0.9701425)
        super(Generator, self).__init__()

        with self.init_scope():
            self.conv1 = L.ConvolutionND(ndim=3, in_channels=1, out_channels=4, ksize=5, pad=get_valid_padding(5), initialW=w)
            self.rrdblock = SkipConnection(layer=2, in_channels=4, out_channels=4, ksize=3, stride=1, pad_type='zero')
            self.conv2 = L.ConvolutionND(ndim=3, in_channels=4, out_channels=1, ksize=5, pad=get_valid_padding(5), initialW=w)

    def forward(self, x):
        h = self.conv2(self.rrdblock(self.conv1(x)))
        return h

    def get_hidden_layer(self, x):
        h1 = self.conv1(x)
        h2 = self.rrdblock.rrdblock(h1)
        return h1, h2


class Discriminator(chainer.Chain):

    def __init__(self, ch=16):
        w = chainer.initializers.Normal(scale=0.02)#Inspired by DCGAN
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.conv1 = SNConvolutionND(ndim=3, in_channels=1, out_channels=ch, ksize=3, stride=1, pad=1, initialW=w)

            self.conv2 = SNConvolutionND(ndim=3, in_channels=ch, out_channels=ch, ksize=3, stride=2, pad=1, initialW=w)

            self.conv3 = SNConvolutionND(ndim=3, in_channels=ch, out_channels=ch*2, ksize=3, stride=1, pad=1, initialW=w)

            self.conv4 = SNConvolutionND(ndim=3, in_channels=ch*2, out_channels=ch*2, ksize=3, stride=2, pad=1, initialW=w)

            self.conv5 = SNConvolutionND(ndim=3, in_channels=ch*2, out_channels=ch*4, ksize=3, stride=1, pad=1, initialW=w)

            self.conv6 = SNConvolutionND(ndim=3, in_channels=ch*4, out_channels=ch*4, ksize=3, stride=2, pad=1, initialW=w)

            self.conv7 = SNConvolutionND(ndim=3, in_channels=ch*4, out_channels=ch*8, ksize=3, stride=1, pad=1, initialW=w)

            self.conv8 = SNConvolutionND(ndim=3, in_channels=ch*8, out_channels=ch*8, ksize=3, stride=2, pad=1, initialW=w)

            self.fc1 = SNLinear(None, ch*16, initialW=w)
            self.fc2 = SNLinear(ch*16, 1, initialW=w)


    def forward(self, x):
        h = F.leaky_relu(self.conv1(x))
        h = F.leaky_relu(self.conv2(h))
        h = F.leaky_relu(self.conv3(h))
        h = F.leaky_relu(self.conv4(h))
        h = F.leaky_relu(self.conv5(h))
        h = F.leaky_relu(self.conv6(h))
        h = F.leaky_relu(self.conv7(h))
        h = F.leaky_relu(self.conv8(h))
        h = F.leaky_relu(self.fc1(h))
        h = self.fc2(h)
        return h
