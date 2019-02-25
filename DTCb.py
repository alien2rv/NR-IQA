# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 12:52:49 2019

@author: RRavela
"""

import pandas as pd
import numpy as np

#prepre live dataset
blur_lcn = np.load('D:\\Databases\\lcn_data\\live\\gblur.npy',allow_pickle=True, fix_imports=True, encoding='ASCII')
awgn_lcn = np.load('D:\\Databases\\lcn_data\\live\\awgn.npy',allow_pickle=True, fix_imports=True, encoding='ASCII')
jpeg_lcn = np.load('D:\\Databases\\lcn_data\\live\\jpeg.npy',allow_pickle=True, fix_imports=True, encoding='ASCII')
jpeg2000_lcn = np.load('D:\\Databases\\lcn_data\\live\\jp2k.npy',allow_pickle=True, fix_imports=True, encoding='ASCII')

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

#train network
from keras.layers import Input,InputLayer, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Sequential,Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
import keras
from keras import backend as K

def create_model(weights_path):
    nb_filters = 50
    nb_conv = 7

    model = Sequential()
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv), input_shape=(32, 32, 3), activation = 'relu', name='conv1') )
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf", name='pool_1'))
    
    model.add(Flatten( name='flatten_1'))
    
    model.add(Dense(units = 800, activation = 'relu',name='FC_1'))
    model.add(Dropout(0.1,name='dropout_1'))

    model.add(Dense(units = 800, activation = 'relu',name='FC_2'))
    model.add(Dropout(0.1, name='dropout_2'))

    model.add(Dense(4, name='fc_3'))
    model.add(Activation('softmax'))
    model.summary()
    
    if weights_path:
        model.load_weights(weights_path)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001))
    return model

model = create_model("D:\\SPIE2019\\coding\\results\\weights\\DTCb_trained.h5")
model.fit(image_train, label_train[:,:4], validation_data=(image_valid, label_valid[:,:4]),
                          epochs=20,verbose=1)

pred = model.predict(image_test)
model.save_weights("D:\\SPIE2019\\coding\\results\\weights\\DTCb_trained.h5")
n = np.argmax(pred,axis=1)
m = np.argmax(label_test[:,:4],axis=1)
from sklearn.metrics import confusion_matrix, accuracy_score
C = confusion_matrix(m,n)
C / C.astype(np.float).sum(axis=1)

import matplotlib.pyplot as plt
from keras.models import Model
layer_outputs = [layer.output for layer in model.layers] 
#layer_outputs = layer_outputs[1:] 
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(image_test[4].reshape(1,32,32,3))
 
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*4,col_size*1))
    plt.axis('off')
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            ax[row,col].axis('off')
            activation_index += 1

plt.axis('off')
plt.imshow(image_test[4][:,:,:]);
plt.show()

display_activation(activations,8, 4,1)
plt.show()