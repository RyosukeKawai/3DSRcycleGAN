# coding:utf-8
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

class SelfAttention2D(chainer.Chain):
    """
    https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
    https://github.com/yuishihara/ChainerTutorial/blob/ec19446537e2ca0613723868f2cc04844cf33108/function_test.py
    """
    def __init__(self, in_channels, out_channels):
        super(SelfAttention2D, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.query_conv = L.Convolution2D(in_channels=in_channels, out_channels=in_channels//8, ksize=1, pad=0, initialW=initializer)
            self.key_conv = L.Convolution2D(in_channels=in_channels, out_channels=in_channels//8, ksize=1, pad=0, initialW=initializer)
            self.value_conv = L.Convolution2D(in_channels=in_channels, out_channels=out_channels, ksize=1, pad=0, initialW=initializer)
            self.gamma = chainer.Parameter(initializer=np.zeros(1, dtype=np.float32))

    def forward(self, x):
        """
        @param x: input feature maps (Batchsize x ch x Height x Width)
        @returns:
            out:
        """
        batchsize, ch, H, W = x.shape
        proj_query = self.query_conv(x).reshape((batchsize, -1, H*W)).transpose((0,2,1)) # (batchsize, HxW, ch)
        proj_key = self.key_conv(x).reshape((batchsize, -1, H*W)) # (batchsize, ch, HxW)
        attention = F.softmax(F.matmul(proj_query, proj_key), axis=1) # (batchsize, ch, HxW)
        proj_value = F.reshape(self.value_conv(x), (batchsize, -1, H*W)) # (batchsize, ch, HxW)

        out = F.matmul(proj_value, attention, transb=True)
        out = F.reshape(out, (batchsize,-1, H, W))

        out = self.gamma*out + x
        return out, attention

class SelfAttention3D(chainer.Chain):
    """
    https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
    https://github.com/yuishihara/ChainerTutorial/blob/ec19446537e2ca0613723868f2cc04844cf33108/function_test.py
    """
    def __init__(self, in_channels, out_channels):
        super(SelfAttention3D, self).__init__()
        initializer = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.query_conv = L.Convolution3D(in_channels=in_channels, out_channels=in_channels//8, ksize=1, pad=0, initialW=initializer)
            self.key_conv = L.Convolution3D(in_channels=in_channels, out_channels=in_channels//8, ksize=1, pad=0, initialW=initializer)
            self.value_conv = L.Convolution3D(in_channels=in_channels, out_channels=out_channels, ksize=1, pad=0, initialW=initializer)
            self.gamma = chainer.Parameter(initializer=np.zeros(1, dtype=np.float32))

    def forward(self, x):
        """
        @param x: input feature maps (Batchsize x ch x Height x Width)
        @returns:
            out:
        """
        batchsize, ch, D, H, W = x.shape
        proj_query = self.query_conv(x).reshape((batchsize, -1, D*H*W)).transpose((0,2,1)) # (batchsize, DxHxW, ch)
        proj_key = self.key_conv(x).reshape((batchsize, -1, D*H*W)) # (batchsize, ch, DxHxW)
        attention = F.softmax(F.matmul(proj_query, proj_key), axis=1) # (batchsize, ch, DxHxW)
        proj_value = F.reshape(self.value_conv(x), (batchsize, -1, D*H*W)) # (batchsize, ch, DxHxW)
        out = F.matmul(proj_value, attention, transb=True)
        out = F.reshape(out, (batchsize, -1, D, H, W))

        out = self.gamma*out + x
        return out, attention
