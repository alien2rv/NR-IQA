# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 23:14:37 2019

@author: rravela
"""

import numpy as np
import os
import cv2
import pandas as pd

folder1 = "D:\\Databases\\single"

X_train = []
def load_train(folder):
#   index = 0
    for filename in os.listdir(folder):
        img = cv2.imread((os.path.join(folder,filename)), cv2.IMREAD_COLOR).astype(np.float32)
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
        X_train.append(img)
        print (filename)
    return X_train

X_train = load_train(folder1)
np.save('D:\\Databases\\single\\single.npy', X_train, allow_pickle=True, fix_imports=True)
