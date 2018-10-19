# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 12:29:45 2018

@author: ravi
"""

import numpy as np
import os
import cv2
import pandas as pd

folder = 'D:\gblur'
dataset = pd.read_csv('blr.csv', header=None)
y = dataset.iloc[:,:].values
X_train = []
Y_train = []

def load_train(folder):
    index = 0
    for filename in os.listdir(folder):
            img = cv2.imread((os.path.join(folder,filename)), cv2.IMREAD_COLOR).astype(np.float32)
            #img = img.transpose((2,0,1))
            (r,c, d) = img.shape
            img_lcn = np.zeros([r,c,d])
            for i in range(3,r-3):
                for j in range(3,c-3):
                    for k in range(0,d):
                        img_patch = img[i-3:i+4, j-3:j+4, k]
                        #mean = sum(sum(img_patch))/49
                        #vari = np.sqrt(sum(sum((img_patch - mean)*(img_patch - mean))))
                        mean = img_patch.mean()
                        vari = img_patch.std()
                        img_lcn[i,j,k] = (img_patch[3,3] - mean)/(vari+10)
            img = img_lcn[3:r-3, 3:c-3, :]
            (r,c,d) = img.shape
            r = int(r/32)*32
            c = int(c/32)*32
            for i in range(0,r,32):
                for j in range(0,c,32):
                    img_patch = img[i:i+32, j:j+32, :]
                    X_train.append(img_patch)
                    Y_train.append(y[index])
            index= index + 1
    return X_train, Y_train

X_train, Y_train = load_train(folder)
x_train = np.asarray(X_train, dtype=np.float32)
y_train = np.asarray(Y_train, dtype=np.float32)

np.save('x_train.npy',x_train, allow_pickle=True, fix_imports=True)
np.save('y_train.npy',y_train, allow_pickle=True, fix_imports=True)