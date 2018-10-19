# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 12:51:19 2018

@author: ravi
"""
import numpy as np

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

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

    if weights_path:
        model.load_weights(weights_path)
    
    model.compile(loss='mean_absolute_error', optimizer=Adam(lr=0.0001))
    return model


model = VGG_16()
plot_model(model, to_file='D:\\cnn\\model.png')
model.fit(x_train,y_train, epochs = 32, batch_size = 32, validation_split=0.3 )
