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
        initializer = chainer.initializers.HeNormal()
        super(ResidualBlock, self).__init__()
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        with self.init_scope():
            self.conv1 = L.ConvolutionND(ndim=3, in_channels=in_channels, out_channels=hidden_channels, ksize=ksize, pad=get_valid_padding(ksize), initialW=initializer)
            self.bn1 = L.BatchNormalization(hidden_channels)
            self.conv2 = L.ConvolutionND(ndim=3, in_channels=hidden_channels, out_channels=out_channels, ksize=ksize, pad=get_valid_padding(ksize), initialW=initializer)
            self.bn2 = L.BatchNormalization(out_channels)

    def forward(self, x):
        h1 = F.leaky_relu(self.bn1(self.conv1(x)))
        h1 = self.bn2(self.conv2(h1))
        return h1+x

class RBList(chainer.ChainList):
    """
    * Residual blocks of srgenerator
    * @param layer # of residual block
    """
    def __init__(self, num_of_layer, in_channels=64, out_channels=64, ksize=3):
        super(RBList, self).__init__()

        for i in range(num_of_layer):
            self.add_link(ResidualBlock(in_channels=in_channels, out_channels=out_channels, ksize=ksize))

    def forward(self, x):
        for f in self.children():
            x = f(x)
        return x

class SkipConnection(chainer.Chain):
    """
    * Skip connection
    """
    def __init__(self, in_channels=64, out_channels=64, ksize=3, num_of_layer=8):
        initializer = chainer.initializers.HeNormal()

        super(SkipConnection, self).__init__()
        with self.init_scope():
            self.rblist = RBList(num_of_layer=num_of_layer, in_channels=in_channels, out_channels=out_channels, ksize=ksize)
            self.conv1 = L.ConvolutionND(ndim=3, in_channels=out_channels, out_channels=out_channels, ksize=ksize, pad=get_valid_padding(ksize), initialW=initializer)
            self.bn1 = L.BatchNormalization(out_channels)
            self.conv_edge1 = L.ConvolutionND(ndim=3, in_channels=out_channels, out_channels=out_channels, ksize=ksize, pad=get_valid_padding(ksize), initialW=initializer)
            self.bn_edge1 = L.BatchNormalization(out_channels)
            self.conv_edge2 = L.ConvolutionND(ndim=3, in_channels=out_channels, out_channels=1, ksize=1, pad=0, initialW=initializer)

    def forward(self, x):
        h = self.bn1(self.conv1(self.rblist(x)))

        return h + x

class Generator_SR(chainer.Chain):
    def __init__(self,ch=64):
        initializer = chainer.initializers.HeNormal()
        super(Generator_SR, self).__init__()

        with self.init_scope():
            self.conv1 = L.ConvolutionND(ndim=3, in_channels=1, out_channels=ch, ksize=5, pad=get_valid_padding(5), initialW=initializer)
            self.resblock = SkipConnection(in_channels=ch, out_channels=ch, ksize=3, num_of_layer=16)
            self.conv2 = L.ConvolutionND(ndim=3, in_channels=ch, out_channels=1, ksize=5, pad=get_valid_padding(5), initialW=initializer)

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x))
        h = self.resblock(h)
        h = self.conv2(h)
        #h32 = cropping(h,32)


        return h#,h32
###############################################################################
# Discriminator
###############################################################################
class Discriminator(chainer.Chain):

    def __init__(self, ch=64):
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
            self.bn4 = L.BatchNormalization(ch * 4)

            self.conv6 = L.ConvolutionND(ndim=3, in_channels=ch *4, out_channels=ch*4, ksize=3, stride=2, pad=1,initialW=w)
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
        h = F.leaky_relu(self.bn6(self.conv7(h)))
        h = F.leaky_relu(self.bn7(self.conv8(h)))
        h = F.leaky_relu(self.fc1(h))
        h = self.fc2(h)
        return h