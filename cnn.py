# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 11:19:50 2018

@author: ravi
"""

import numpy as np

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Lambda
from keras.layers.convolutional import Conv2D, ZeroPadding2D, MaxPooling2D
from keras.optimizers import RMSprop, Adam, Adadelta
from keras import backend as K


def min_max_pool2d(x):
    max_x =  K.pool2d(x, pool_size=(1, 1))
    min_x = -K.pool2d(-x, pool_size=(1, 1))
    return K.concatenate([max_x, min_x], axis=1) # concatenate on channel

def min_max_pool2d_output_shape(input_shape):
    shape = list(input_shape)
    shape[1] *= 2
    return tuple(shape)

def create_model():
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

    model.add(Dense(1, name='fc_3'))
    model.add(Activation('sigmoid'))

    model.compile(loss='mean_squared_error', optimizer=Adadelta())
    return model

model = create_model()
plot_model(model, to_file='D:\\cnn\\model.png', show_shapes = True)
model.fit(x_train,y_train, epochs = 2, batch_size = 32)
y_valid = model.predict(X_test)