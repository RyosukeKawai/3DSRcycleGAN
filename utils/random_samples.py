# coding:utf-8
"""
https://github.com/pfnet-research/sngan_projection/blob/master/source/miscs/random_samples.py
"""
import numpy as np

def sample_continuous(dim, batchsize, distribution='normal', xp=np):
    if distribution == "normal":
        return xp.random.randn(batchsize, dim) \
            .astype(xp.float32)
    elif distribution == "uniform":
        return xp.random.uniform(-1, 1, (batchsize, dim)) \
            .astype(xp.float32)
    else:
        raise NotImplementedError
