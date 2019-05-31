# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 01:17:59 2019

@author: rravela
"""
import pandas as pd
import numpy as np

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

from keras.layers import Input,InputLayer, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Sequential,Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
import keras
from keras import backend as K
from keras.utils import plot_model

#from DTC import image_split_d,dtc_VG
from DTC import image_split,dtc_resnet
from gblur import gblur_resnet
from jpeg import jpeg_resnet
from jp2k import jp2k_resnet
from awgn import awgn_resnet

#prepre live dataset
blur_lcn = np.load('D:\\Databases\\lcn_data\\tid\\gblur.npy',allow_pickle=True, fix_imports=True, encoding='ASCII')
awgn_lcn = np.load('D:\\Databases\\lcn_data\\tid\\awgn.npy',allow_pickle=True, fix_imports=True, encoding='ASCII')
jpeg_lcn = np.load('D:\\Databases\\lcn_data\\tid\\jpeg.npy',allow_pickle=True, fix_imports=True, encoding='ASCII')
jpeg2000_lcn = np.load('D:\\Databases\\lcn_data\\tid\\jp2k.npy',allow_pickle=True, fix_imports=True, encoding='ASCII')

image =np.concatenate((jpeg_lcn,jpeg2000_lcn,blur_lcn,awgn_lcn))
image_test = np.concatenate((image[80:100],image[180:200],image[280:300],image[380:400]))

dataset = pd.read_csv('D:\\Databases\\lcn_data\\tid\\labels\\jpeg.csv')
b = dataset.iloc[:].values
label_jpeg = np.zeros((len(b),3))
label_jpeg[:,:-1] = b
dataset = pd.read_csv('D:\\Databases\\lcn_data\\tid\\labels\\jp2k.csv')
b = dataset.iloc[:].values
label_jp2k = np.zeros((len(b),3))
label_jp2k[:,:-1] = b
dataset = pd.read_csv('D:\\Databases\\lcn_data\\tid\\labels\\gblur.csv')
b = dataset.iloc[:].values
label_gblur = np.zeros((len(b),3))
label_gblur[:,:-1] = b
dataset = pd.read_csv('D:\\Databases\\lcn_data\\tid\\labels\\awgn.csv')
b = dataset.iloc[:].values
label_wn = np.zeros((len(b),3))
label_wn[:,:-1] = b
label = np.concatenate((label_jpeg,label_jp2k,label_gblur,label_wn))
label_test = np.concatenate((label[80:100],label[180:200],label[280:300],label[380:400]))

#load image split
X_test,Y_test = image_split(image_test,32,label_test)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

#blur
model_blur = gblur_resnet("D:\\thesis\\coding\\resnets\\model\\gblur.h5")
pred_blur = model_blur.predict(X_test)

# awgn
model_awgn = awgn_resnet("D:\\thesis\\coding\\resnets\\model\\awgn.h5")
pred_awgn = model_awgn.predict(X_test)

# jpeg
model_jpeg = jpeg_resnet("D:\\thesis\\coding\\resnets\\model\\jpeg.h5")
pred_jpeg = model_jpeg.predict(X_test)

# jpeg
model_jp2k = jp2k_resnet("D:\\thesis\\coding\\resnets\\model\\jp2k.h5")
pred_jp2k = model_jp2k.predict(X_test)
#dtc
mdel_dtc = dtc_resnet("D:\\thesis\\coding\\resnets\\model\\DTC.h5")
DTC_data = mdel_dtc.predict(X_test)

pred = ((DTC_data[:,0]*pred_jpeg[:,0])+(DTC_data[:,1]*pred_jp2k[:,0])+(DTC_data[:,2]*pred_blur[:,0])+(DTC_data[:,3]*pred_awgn[:,0]))
pred = pred.reshape(13200,1)
Y_pred = []
s=0
count = 0
for i in range(1,len(Y_test),1):
    if (Y_test[i-1,2] == Y_test[i,2]):
        s = s+pred[i,0]
        count = count+1
    else:
        mean = s/count
        Y_pred.append(mean)
        s=0
        count=0
Y_pred.append(s/count)

from scipy import stats

Y_pred = 1- np.array(Y_pred)
PLCC_d = stats.pearsonr(label_test[:,0], Y_pred)
print('all',PLCC_d)
SRCC_d = stats.spearmanr(label_test[:,0], Y_pred)
print(SRCC_d)

PLCC_d = stats.pearsonr(label_test[:20,0], Y_pred[:20])
print('jpg',PLCC_d)
SRCC_d = stats.spearmanr(label_test[:20,0], Y_pred[:20])
print(SRCC_d)

PLCC_d = stats.pearsonr(label_test[20:40,0], Y_pred[20:40])
print('jp2k',PLCC_d)
SRCC_d = stats.spearmanr(label_test[20:40,0], Y_pred[20:40])
print(SRCC_d)

PLCC_d = stats.pearsonr(label_test[40:60,0], Y_pred[40:60])
print('blr',PLCC_d)
SRCC_d = stats.spearmanr(label_test[40:60,0], Y_pred[40:60])
print(SRCC_d)

PLCC_d = stats.pearsonr(label_test[60:,0], Y_pred[60:])
print('wn',PLCC_d)
SRCC_d = stats.spearmanr(label_test[60:,0], Y_pred[60:])
print(SRCC_d)

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(label_test[:,0],Y_pred)