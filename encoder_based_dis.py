#default
import sys, os, time, math
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
import chainer.link_hooks

###############################################################################
# Discriminator
###############################################################################
class Discriminator(chainer.Chain):

    def __init__(self, ch=64):
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


            self.conv8 = L.ConvolutionND(ndim=3, in_channels=ch*8, out_channels=ch*8,ksize=3, stride=2, pad=1, initialW=w).add_hook(chainer.link_hooks.SpectralNormalization())


            self.fc1 = L.Linear(None, ch*16, initialW=w)
            self.fc2 = L.Linear(ch*16, 1, initialW=w)

    def forward(self, x):
        h = F.leaky_relu(self.conv1(x))
        h = F.leaky_relu(self.conv2(h))
        h = F.leaky_relu(self.conv3(h))
        h = F.leaky_relu(self.conv4(h))
        h = F.leaky_relu(self.conv5(h))
        h = F.leaky_relu(self.conv7(h))
        h = F.leaky_relu(self.conv8(h))
        h = F.leaky_relu(self.fc1(h))
        h = self.fc2(h)
        return h
