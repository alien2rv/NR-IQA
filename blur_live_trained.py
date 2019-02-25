# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 11:41:17 2019

@author: RRavela
"""

import numpy as np
import pandas as pd

blur_lcn = np.load('D:\\Databases\\lcn_data\\live\\gblur.npy',allow_pickle=True, fix_imports=True, encoding='ASCII')
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
        [r,c,d] = img.shape
        r = int(r/blk_size)*blk_size
        c = int(c/blk_size)*blk_size
        for i in range(0,r,blk_size):
            for j in range(0,c,blk_size):
                img_patch = img[i:i+blk_size, j:j+blk_size, :]
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

#model for blur
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
from keras.utils import plot_model

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(32,32,3)))
    model.add(Conv2D(32,kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32,kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64,kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64,kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128,kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128,kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256,kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256,kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(units = 512, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))
    model.summary()

    if weights_path:
        model.load_weights(weights_path)
    
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))
    return model

model = VGG_16("D:\\SPIE2019\\coding\\results\\weights\\blur_trained.h5")
model.fit(X_train, Y_train[:,:1], validation_data=(X_valid, Y_valid[:,:1]),
                          epochs=20,verbose=1)

pred = model.predict(X_test)
model.save_weights("D:\\SPIE2019\\coding\\results\\weights\\blur_trained.h5")
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
np.save("D:\\SPIE2019\\coding\\results\\live\\blur_predicted.npy", Y_pred, allow_pickle=True, fix_imports=True)
np.save("D:\\SPIE2019\\coding\\results\\live\\blur_test.npy", Y_test, allow_pickle=True, fix_imports=True)

import matplotlib.pyplot as plt
plt.scatter(label_test[:,0], Y_pred)


# visualising all the convolutional layers
from keras.models import Model
layer_outputs = [layer.output for layer in model.layers] 
layer_outputs = layer_outputs[1:] 
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(X_test[4].reshape(1,32,32,3))
 
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*2.5))
    plt.axis('off')
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            ax[row,col].axis('off')
            activation_index += 1

plt.axis('off')
plt.imshow(X_test[4][:,:,:]);
plt.show()

display_activation(activations,8, 8,5)
plt.show()