# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:51:58 2019

@author: RRavela
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# Any results you write to the current directory are saved as output.
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Convolution2D, Input,Activation, ZeroPadding2D, MaxPooling2D, Flatten, merge
from keras.optimizers import SGD, Adam
from keras.objectives import sparse_categorical_crossentropy as scc
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from keras.utils import plot_model

#prepre live dataset
blur_lcn = np.load('D:\\Databases\\GCN\\gblur.npy',allow_pickle=True, fix_imports=True, encoding='ASCII')
awgn_lcn = np.load('D:\\Databases\\GCN\\awgn.npy',allow_pickle=True, fix_imports=True, encoding='ASCII')
jpeg_lcn = np.load('D:\\Databases\\GCN\\jpeg.npy',allow_pickle=True, fix_imports=True, encoding='ASCII')
jpeg2000_lcn = np.load('D:\\Databases\\GCN\\jp2k.npy',allow_pickle=True, fix_imports=True, encoding='ASCII')

dataset = pd.read_csv('D:\\Databases\\lcn_data\\live\\labels\\DTC_live.csv')
b = dataset.iloc[:].values
label = np.zeros((len(b),len(b[0])+1))
label[:,:-1] = b

label_train = np.concatenate((label[:120],label[175:293],label[344:444],label[489:589]))
label_test = np.concatenate((label[120:151],label[293:321],label[444:469],label[589:614]))
label_valid = np.concatenate((label[151:175],label[321:344],label[469:489],label[614:634]))

image =np.concatenate((jpeg_lcn,jpeg2000_lcn,blur_lcn,awgn_lcn))
blur_lcn = []
jpeg_lcn = []
jpeg2000_lcn = []
awgn_lcn = []
image_train = np.concatenate((image[:120],image[175:293],image[344:444],image[489:589]))
image_test = np.concatenate((image[120:151],image[293:321],image[444:469],image[589:614]))
image_valid = np.concatenate((image[151:175],image[321:344],image[469:489],image[614:634]))
image = []

#training the model
def image_split(k,blk_size,label):
    X_train = []
    Y_train = []
    for a in range(len(k)):
        img = k[a]
        label[a,5]=a
        [r,c,d] = img.shape
        r = int(r/blk_size)*blk_size
        c = int(c/blk_size)*blk_size
        for i in range(0,r,blk_size):
            for j in range(0,c,blk_size):
                img_patch = img[i:i+blk_size, j:j+blk_size, :]
                X_train.append(img_patch)
                Y_train.append(label[a])
    return X_train, Y_train

image_train,label_train = image_split(image_train,32,label_train)
image_train = np.array(image_train)
label_train = np.array(label_train)

image_test,label_test = image_split(image_test,32,label_test)
image_test = np.array(image_test)
label_test = np.array(label_test)

image_valid,label_valid = image_split(image_valid,32,label_valid)
image_valid = np.array(image_valid)
label_valid = np.array(label_valid)

def get_resnet():
    # In order to make things less confusing, all layers have been declared first, and then used
    
    # declaration of layers
    input_img = Input((32, 32,3), name='input_layer')
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

    final = Dense(4, activation='softmax', init='he_uniform', name='classifier')
    
    # declaration completed
    
    first = zeroPad1(input_img)
    second = layer1(first)
    second = BatchNormalization(name='major_bn')(second)
    second = Activation('relu', name='major_act')(second)

    third = zeroPad2(second)
    third = layer2(third)
    third = BatchNormalization(name='l1_bn')(third)
    third = Activation('relu', name='l1_act')(third)

    third = zeroPad3(third)
    third = layer3(third)
    third = BatchNormalization(name='l1_bn2')(third)
    third = Activation('relu', name='l1_act2')(third)


    res = keras.layers.Add(name='res')([third, second])


    first2 = zeroPad1_2(res)
    second2 = layer1_2(first2)
    second2 = BatchNormalization(name='major_bn2')(second2)
    second2 = Activation('relu', name='major_act2')(second2)


    third2 = zeroPad2_2(second2)
    third2 = layer2_2(third2)
    third2 = BatchNormalization(name='l2_bn')(third2)
    third2 = Activation('relu', name='l2_act')(third2)

    third2 = zeroPad3_2(third2)
    third2 = layer3_2(third2)
    third2 = BatchNormalization(name='l2_bn2')(third2)
    third2 = Activation('relu', name='l2_act2')(third2)

    res2 = keras.layers.Add(name='res2')([third2, second2])

    res2 = Flatten()(res2)

    res2 = layer4(res2)
    res2 = Dropout(0.4, name='dropout1')(res2)
    res2 = layer5(res2)
    res2 = Dropout(0.4, name='dropout2')(res2)
    res2 = final(res2)
    model = Model(input=input_img, output=res2)
    
    
    #sgd = SGD(decay=0., lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    return model

res = get_resnet()
res.summary()
plot_model(res, to_file='D:\\cnn\\model10.png', show_shapes = True)
res.fit(image_train, label_train[:,:4], validation_data=(image_valid, label_valid[:,:4]),
                          epochs=20,verbose=1)
pred = res.predict(image_test)
res.save_weights("D:\\thesis\\coding\\resnets\\DTC_trained_gcn.h5")
n = np.argmax(pred,axis=1)
m = np.argmax(label_test[:,:4],axis=1)
from sklearn.metrics import confusion_matrix, accuracy_score
C = confusion_matrix(m,n)
C / C.astype(np.float).sum(axis=1)
K = accuracy_score(m,n)
