# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:38:28 2019

@author: rravela
"""
import numpy as np
import pandas as pd

blur_lcn = np.load('D:\\Databases\\Gray_ scale\\gblur.npy',allow_pickle=True, fix_imports=True, encoding='ASCII')
dataset = pd.read_csv('D:\\Databases\\lcn_data\\live\\labels\\live_gblur_sorted.csv')
b = dataset.iloc[:].values
label = np.zeros((len(b),3))
label[:,:-1] = b

X_train = blur_lcn[:95]
X_valid = blur_lcn[95:120]
X_test = blur_lcn[120:145]

Y_train = label[:95]
Y_valid = label[95:120]
label_test = label[120:145]

def image_split(k,blk_size,label):
    X_train = []
    Y_train = []
    for a in range(len(k)):
        img = k[a]
        label[a,2]=a
        [r,c] = img.shape
        r = int(r/blk_size)*blk_size
        c = int(c/blk_size)*blk_size
        for i in range(0,r,blk_size):
            for j in range(0,c,blk_size):
                img_patch = img[i:i+blk_size, j:j+blk_size]
                X_train.append(img_patch)
                Y_train.append(label[a])
    return X_train, Y_train

X_train,Y_train = image_split(X_train,32,Y_train)
X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_test,Y_test = image_split(X_test,32,label_test)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

X_valid,Y_valid = image_split(X_valid,32,Y_valid)
X_valid = np.array(X_valid)
Y_valid = np.array(Y_valid)

import keras
from keras.models import Model, Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Convolution2D, Input,Activation, ZeroPadding2D, MaxPooling2D, Flatten, merge
from keras.optimizers import SGD, Adam
from keras.objectives import sparse_categorical_crossentropy as scc
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from keras.utils import plot_model

def get_resnet():
    # In order to make things less confusing, all layers have been declared first, and then used
    
    # declaration of layers
    input_img = Input((32,32,3), name='input_layer')
    zeroPad1 = ZeroPadding2D((1,1), name='zeroPad1', dim_ordering='tf')
    zeroPad1_2 = ZeroPadding2D((1,1), name='zeroPad1_2', dim_ordering='tf')
    layer1 = Convolution2D(32, 3, 3, subsample=(2, 2), init='he_uniform', name='major_conv', dim_ordering='tf')
    layer1_2 = Convolution2D(64, 3, 3, subsample=(2, 2), init='he_uniform', name='major_conv2', dim_ordering='tf')
    zeroPad2 = ZeroPadding2D((1,1), name='zeroPad2', dim_ordering='tf')
    zeroPad2_2 = ZeroPadding2D((1,1), name='zeroPad2_2', dim_ordering='tf')
    layer2 = Convolution2D(32, 3, 3, subsample=(1,1), init='he_uniform', name='l1_conv', dim_ordering='tf')
    layer2_2 = Convolution2D(64, 3, 3, subsample=(1,1), init='he_uniform', name='l1_conv2', dim_ordering='tf')


    zeroPad3 = ZeroPadding2D((1,1), name='zeroPad3', dim_ordering='tf')
    zeroPad3_2 = ZeroPadding2D((1,1), name='zeroPad3_2', dim_ordering='tf')
    layer3 = Convolution2D(32, 3, 3, subsample=(1, 1), init='he_uniform', name='l2_conv', dim_ordering='tf')
    layer3_2 = Convolution2D(64, 3, 3, subsample=(1, 1), init='he_uniform', name='l2_conv2', dim_ordering='tf')

    layer4 = Dense(512, activation='relu', init='he_uniform', name='dense1')
    layer5 = Dense(256, activation='relu', init='he_uniform', name='dense2')

    final = Dense(1, activation='linear', init='he_uniform', name='regressor')
    
    # declaration completed
    
    first = zeroPad1(input_img)
    second = layer1(first)
    second = BatchNormalization(name='major_bn')(second)
    second = Activation('tanh', name='major_act')(second)

    third = zeroPad2(second)
    third = layer2(third)
    third = BatchNormalization(name='l1_bn')(third)
    third = Activation('tanh', name='l1_act')(third)

    third = zeroPad3(third)
    third = layer3(third)
    third = BatchNormalization(name='l1_bn2')(third)
    third = Activation('tanh', name='l1_act2')(third)


    res = keras.layers.Add(name='res')([third, second])


    first2 = zeroPad1_2(res)
    second2 = layer1_2(first2)
    second2 = BatchNormalization(name='major_bn2')(second2)
    second2 = Activation('tanh', name='major_act2')(second2)


    third2 = zeroPad2_2(second2)
    third2 = layer2_2(third2)
    third2 = BatchNormalization(name='l2_bn')(third2)
    third2 = Activation('tanh', name='l2_act')(third2)

    third2 = zeroPad3_2(third2)
    third2 = layer3_2(third2)
    third2 = BatchNormalization(name='l2_bn2')(third2)
    third2 = Activation('tanh', name='l2_act2')(third2)

    res2 = keras.layers.Add(name='res2')([third2, second2])

    res2 = Flatten()(res2)

    res2 = layer4(res2)
    res2 = Dropout(0.4, name='dropout1')(res2)
    res2 = layer5(res2)
    res2 = Dropout(0.4, name='dropout2')(res2)
    res2 = final(res2)
    model = Model(input=input_img, output=res2)
    
    
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-4, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer= adam)
    return model

res = get_resnet()
res.summary()
res.fit(X_train, (Y_train[:,:1]*0.01), validation_data=(X_valid, (Y_valid[:,:1]*0.01)),
                          epochs=20,verbose=1)
pred = res.predict(X_test)
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

PLCC = stats.pearsonr(label_test[:,0], Y_pred)
print(PLCC)
SRCC = stats.spearmanr(label_test[:,0], Y_pred)
print(SRCC)

#res.save_weights("D:\\thesis\\coding\\resnets\\blur_trained_gray.h5")