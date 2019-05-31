# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 23:05:26 2019

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

from DTC import image_split_d,dtc_model
from expertIQA import gblur_model,image_split,awgn_model,jpeg_model,jp2k_model

#prepre live dataset
blur_lcn = np.load('D:\\Databases\\lcn_data\\live\\gblur.npy',allow_pickle=True, fix_imports=True, encoding='ASCII')
awgn_lcn = np.load('D:\\Databases\\lcn_data\\live\\awgn.npy',allow_pickle=True, fix_imports=True, encoding='ASCII')
jpeg_lcn = np.load('D:\\Databases\\lcn_data\\live\\jpeg.npy',allow_pickle=True, fix_imports=True, encoding='ASCII')
jpeg2000_lcn = np.load('D:\\Databases\\lcn_data\\live\\jp2k.npy',allow_pickle=True, fix_imports=True, encoding='ASCII')

image =np.concatenate((jpeg_lcn,jpeg2000_lcn,blur_lcn,awgn_lcn))
image_test = np.concatenate((image[120:151],image[293:321],image[444:469],image[589:614]))

dataset = pd.read_csv('D:\\Databases\\lcn_data\\live\\labels\\live_jpeg_sorted.csv')
b = dataset.iloc[:].values
label_jpeg = np.zeros((len(b),3))
label_jpeg[:,:-1] = b
dataset = pd.read_csv('D:\\Databases\\lcn_data\\live\\labels\\live_jp2k_sorted.csv')
b = dataset.iloc[:].values
label_jp2k = np.zeros((len(b),3))
label_jp2k[:,:-1] = b
dataset = pd.read_csv('D:\\Databases\\lcn_data\\live\\labels\\live_gblur_sorted.csv')
b = dataset.iloc[:].values
label_gblur = np.zeros((len(b),3))
label_gblur[:,:-1] = b
dataset = pd.read_csv('D:\\Databases\\lcn_data\\live\\labels\\live_wn_sorted.csv')
b = dataset.iloc[:].values
label_wn = np.zeros((len(b),3))
label_wn[:,:-1] = b
label = np.concatenate((label_jpeg,label_jp2k,label_gblur,label_wn))
label_test = np.concatenate((label[120:151],label[293:321],label[444:469],label[589:614]))

#load image split
X_test,Y_test = image_split(image_test,32,label_test)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

#blur
model_blur = gblur_model("D:\\thesis\\coding\\shallownets\\gblur.h5")
pred_blur = model_blur.predict(X_test)

# awgn
model_awgn = awgn_model("D:\\thesis\\coding\\shallownets\\awgn.h5")
pred_awgn = model_awgn.predict(X_test)

# jpeg
model_jpeg = jpeg_model("D:\\thesis\\coding\\shallownets\\jpeg.h5")
pred_jpeg = model_jpeg.predict(X_test)

# jpeg
model_jp2k = jp2k_model("D:\\thesis\\coding\\shallownets\\jp2k.h5")
pred_jp2k = model_jp2k.predict(X_test)

DT = dtc_model("D:\\thesis\\coding\\shallownets\\DTC.h5")
DTC_data = DT.predict(X_test)

pred = ((DTC_data[:,0]*pred_jpeg[:,0])+(DTC_data[:,1]*pred_jp2k[:,0])+(DTC_data[:,2]*pred_blur[:,0])+(DTC_data[:,3]*pred_awgn[:,0]))
pred = pred.reshape(33705,1)
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

PLCC_d = stats.pearsonr(label_test[:,0], Y_pred)
print('all',PLCC_d)
SRCC_d = stats.spearmanr(label_test[:,0], Y_pred)
print(SRCC_d)

PLCC_d = stats.pearsonr(label_test[:31,0], Y_pred[:31])
print('jpg',PLCC_d)
SRCC_d = stats.spearmanr(label_test[:31,0], Y_pred[:31])
print(SRCC_d)

PLCC_d = stats.pearsonr(label_test[31:59,0], Y_pred[31:59])
print('jp2k',PLCC_d)
SRCC_d = stats.spearmanr(label_test[31:59,0], Y_pred[31:59])
print(SRCC_d)

PLCC_d = stats.pearsonr(label_test[59:84,0], Y_pred[59:84])
print('blr',PLCC_d)
SRCC_d = stats.spearmanr(label_test[59:84,0], Y_pred[59:84])
print(SRCC_d)

PLCC_d = stats.pearsonr(label_test[84:,0], Y_pred[84:])
print('wn',PLCC_d)
SRCC_d = stats.spearmanr(label_test[84:,0], Y_pred[84:])
print(SRCC_d)

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(label_test[:,0],Y_pred)

# visualising all the convolutional layers
import matplotlib.pyplot as plt
from keras.models import Model
layer_outputs = [layer.output for layer in DT.layers] 
layer_outputs = layer_outputs[1:] 
activation_model = Model(inputs=DT.input, outputs=layer_outputs)
activations = activation_model.predict(X_test[21004].reshape(1,32,32,3))
 
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*3,col_size*1))
    plt.axis('off')
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            ax[row,col].axis('off')
            activation_index += 1

plt.axis('off')
plt.imshow(X_test[4][:,:,:]);
plt.show()

display_activation(activations,8,4 ,0)
plt.show()
