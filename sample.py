#ただの落書き

import numpy as np


# load image data
print("train.mhd load")
train =np.zeros([1240,1210,3600],dtype=float)
print("train.mhd load done")

print(train.shape)

train1,train2 =np.split(train,[800],1)

print(train1.shape)
print(train2.shape)