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

def get_valid_padding(kernel_size):
    """
    @param: kernel_size, kernel size of conv
    @return: spatial padding width
    'https://github.com/xinntao/ESRGAN/blob/master/block.py'
    """
    padding = (kernel_size - 1) // 2
    return padding

class BottleNeck(chainer.Chain):
    """
    * single residual block
    """
    def __init__(self, in_channels=64, hidden_channels=None, out_channels=64, ksize=3):
        w = chainer.initializers.HeNormal(scale=0.9701425)
        super(BottleNeck, self).__init__()
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        with self.init_scope():
            self.conv1 = L.ConvolutionND(ndim=3, in_channels=in_channels, out_channels=hidden_channels, ksize=ksize, pad=get_valid_padding(ksize), initialW=w)
            self.bn1 = L.BatchNormalization(hidden_channels)
            self.prelu = L.PReLU()
            self.conv2 = L.ConvolutionND(ndim=3, in_channels=hidden_channels, out_channels=out_channels, ksize=ksize, pad=get_valid_padding(ksize), initialW=w)
            self.bn2 = L.BatchNormalization(out_channels)

    def forward(self, x):
        h1 = self.prelu(self.bn1(self.conv1(x)))
        h1 = self.bn2(self.conv2(h1))

        return (h1 + x)

class BlockA(chainer.ChainList):
    """
    * Residual blocks of srgenerator
    * @param layer # of residual block
    """
    def __init__(self, num_of_layer, in_channels=64, out_channels=64, ksize=3):
        super(BlockA, self).__init__()

        for i in range(num_of_layer):
            self.add_link(BottleNeck(in_channels=in_channels, out_channels=out_channels, ksize=ksize))

    def forward(self, x):
        for f in self.children():
            x = f(x)
        return x

class BlockB(chainer.Chain):
    """
    * Residual blocks and skip connection
    """
    def __init__(self, in_channels=64, out_channels=64, ksize=3, num_of_layer=16):
        w = chainer.initializers.HeNormal(scale=0.9701425)
        super(BlockB, self).__init__()

        with self.init_scope():
            self.res = BlockA(num_of_layer = num_of_layer, in_channels=in_channels, out_channels=out_channels, ksize=ksize)
            self.conv1 = L.ConvolutionND(ndim=3, in_channels=out_channels, out_channels=out_channels, ksize=ksize, pad=get_valid_padding(ksize), initialW=w)
            self.bn1 = L.BatchNormalization(out_channels)

    def forward(self, x):
        h = self.bn1(self.conv1(self.res(x)))
        return (h+x)

class Generator(chainer.Chain):

    def __init__(self):
        w = chainer.initializers.HeNormal(scale=0.9701425)
        super(Generator, self).__init__()

        with self.init_scope():
            self.conv1 = L.ConvolutionND(ndim=3, in_channels=1, out_channels=32, ksize=5, pad=2, initialW=w)
            self.prelu1 = L.PReLU()
            self.resblock = BlockB(in_channels=32, out_channels=32, ksize=3, num_of_layer=8)
            self.conv2 = L.ConvolutionND(ndim=3, in_channels=32, out_channels=1, ksize=5, pad=2, initialW=w)

    def forward(self, x):
        h = self.prelu1(self.conv1(x))
        h = self.resblock(h)
        h = self.conv2(h)

        return h

class Discriminator(chainer.Chain):

    def __init__(self, ch=16):
        w = chainer.initializers.Normal(scale=0.02)#Inspired by DCGAN
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.conv1 = L.ConvolutionND(ndim=3, in_channels=1, out_channels=ch, ksize=3, stride=1, pad=1, initialW=w)

            self.conv2 = L.ConvolutionND(ndim=3, in_channels=ch, out_channels=ch, ksize=3, stride=2, pad=1, initialW=w)
            self.bn1 = L.BatchNormalization(ch)

            self.conv3 = L.ConvolutionND(ndim=3, in_channels=ch, out_channels=ch*2, ksize=3, stride=1, pad=1, initialW=w)
            self.bn2 = L.BatchNormalization(ch*2)

            self.conv4 = L.ConvolutionND(ndim=3, in_channels=ch*2, out_channels=ch*2, ksize=3, stride=2, pad=1, initialW=w)
            self.bn3 = L.BatchNormalization(ch*2)

            self.conv5 = L.ConvolutionND(ndim=3, in_channels=ch*2, out_channels=ch*4, ksize=3, stride=1, pad=1, initialW=w)
            self.bn4 = L.BatchNormalization(ch*4)

            self.conv6 = L.ConvolutionND(ndim=3, in_channels=ch*4, out_channels=ch*4, ksize=3, stride=2, pad=1, initialW=w)
            self.bn5 = L.BatchNormalization(ch*4)

            self.conv7 = L.ConvolutionND(ndim=3, in_channels=ch*4, out_channels=ch*8, ksize=3, stride=1, pad=1, initialW=w)
            self.bn6 = L.BatchNormalization(ch*8)

            self.conv8 = L.ConvolutionND(ndim=3, in_channels=ch*8, out_channels=ch*8, ksize=3, stride=2, pad=1, initialW=w)
            self.bn7 = L.BatchNormalization(ch*8)

            self.fc1 = L.Linear(None, ch*16, initialW=w)
            self.fc2 = L.Linear(ch*16, 1, initialW=w)


    def forward(self, x):
        h = F.leaky_relu(self.conv1(x))
        h = F.leaky_relu(self.bn1(self.conv2(h)))
        h = F.leaky_relu(self.bn2(self.conv3(h)))
        h = F.leaky_relu(self.bn3(self.conv4(h)))
        h = F.leaky_relu(self.bn4(self.conv5(h)))
        h = F.leaky_relu(self.bn5(self.conv6(h)))
        h = F.leaky_relu(self.bn6(self.conv7(h)))
        h = F.leaky_relu(self.bn7(self.conv8(h)))
        h = F.leaky_relu(self.fc1(h))
        h = self.fc2(h)
        return h
