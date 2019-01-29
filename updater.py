﻿#coding:utf-8
"""
* @auther ryosuke
* reference source :https://github.com/zEttOn86/3D-SRGAN
"""

#default
import os, time, sys, copy
import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer.dataset import concat_examples
from chainer import Variable
import random

def cropping(input, ref):
    ref_map=np.zeros((ref,ref,ref))
    rZ, rY, rX =ref_map.shape
    _, _, iZ, iY, iX = input.shape
    edgeZ, edgeY, edgeX = (iZ - rZ)//2, (iY - rY)//2, (iX - rX)//2
    edgeZZ, edgeYY, edgeXX = iZ-edgeZ, iY-edgeY, iX-edgeX

    _, X, _ = F.split_axis(input, (edgeX, edgeXX), axis=4)
    _, X, _ = F.split_axis(X, (edgeY, edgeYY), axis=3)
    _, X, _ = F.split_axis(X, (edgeZ, edgeZZ), axis=2)

    return X

class CinCGANUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen_SR,self.gen2,self.disY= kwargs.pop("models")
        super(CinCGANUpdater, self).__init__(*args, **kwargs)

        self._lambda_A = 10 #戸澤さんの経験則：Loss_advの値がLoss_cycの1/10ぐらいになるといい感じ
        self._lambda_B = 10


    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_real)) / batchsize
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1+L2
        chainer.report({'loss':loss, 'L1':L1, 'L2':L2}, dis)
        return loss

    def loss_gen_SR(self, gen,x,y,x_fake_y,y_fake_x,y_fake):
        batchsize = len(y_fake)
        L1 = F.sum(F.softplus(-y_fake)) / batchsize
        L2 = self._lambda_A*F.mean_absolute_error(x,x_fake_y)#loss cycLHL
        L3 = self._lambda_B*F.mean_absolute_error(y,y_fake_x)#loss cycHLH
        loss_cyc=L2+L3
        loss = L1 + loss_cyc
        chainer.report({'loss': loss, 'L1': L1, 'L2': L2,'L3': L3,'Loss_cyc':loss_cyc }, gen)
        return loss




    def update_core(self):
        gen1_optimizer = self.get_optimizer("gen")#load optimizer called "gen"
        gen2_optimizer = self.get_optimizer("gen2")
        disY_optimizer = self.get_optimizer("disY")
        batch = self.get_iterator("main").next()#iterator
        #x=lr y=hr
        x, y = self.converter(batch, self.device)

        gen_SR=self.gen_SR
        gen2=self.gen2
        disY=self.disY

        y_fake , y_fake_32 = gen_SR(x)# G(x)
        x_fake_y , x_fake_y_32 = gen2(y_fake)

        x_fake , x_fake_32 = gen2(y)
        y_fake_x , y_fake_x_32=gen_SR(x_fake)

        # 32+x →　32
        y=cropping(y,32)
        x=cropping(x,32)

        #discriminator
        y_real_dis=disY(y)
        y_fake_dis=disY(y_fake_32)

        disY_optimizer.update(self.loss_dis, disY, y_fake_dis, y_real_dis)
        gen1_optimizer.update(self.loss_gen_SR, gen_SR,x,y,)#
        gen2_optimizer.update(self.loss_gen_SR, gen2, y_fake, y, y_fake_dis)


class preUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen1 ,self.gen2= kwargs.pop("models")#self.gen1 , self.gen2= kwargs.pop("models")
        super(preUpdater, self).__init__(*args, **kwargs)
        self._lambda_A = 1 #preは１
        self._lambda_B = 1

    def loss_cyc_l1(self, fake, real):
        return F.mean_absolute_error(fake, real)

    def update_core(self):

        #optimizer設置
        gen1_optimizer=self.get_optimizer("gen")
        gen2_optimizer = self.get_optimizer("gen2")
        gen1=self.gen1
        gen2 = self.gen2
        batch = self.get_iterator("main").next()#iterator

        #x-PseudoLR y-MicroCT
        x, y = self.converter(batch, self.device)

        y_fake= gen1(x)
        x_fake_y = gen2(y_fake)

        x_fake = gen2(y)
        y_fake_x = gen1(x_fake)

        # # Generator_loss
        # x = cropping(x, 32)#cropping
        # y = cropping(y, 32)  # cropping

        loss_cyc_LHL = self._lambda_A*self.loss_cyc_l1(x, x_fake_y)

        loss_cyc_HLH = self._lambda_B*self.loss_cyc_l1(y, y_fake_x)

        loss_gen=loss_cyc_LHL + loss_cyc_HLH

        # gen update
        gen1.cleargrads()
        gen2.cleargrads()
        loss_gen.backward()
        gen1_optimizer.update()
        gen2_optimizer.update()

        chainer.report({'loss_gen_pre': loss_gen})
        chainer.report({'loss_cyc_LHL': loss_cyc_LHL})
        chainer.report({'loss_cyc_HLH': loss_cyc_HLH})












