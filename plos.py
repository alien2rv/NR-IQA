# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 15:32:38 2018

@author: ravi
"""
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

import keras
from keras.optimizers import Adam, SGD
from keras.models import Model
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
from keras.layers.core import Layer
from keras.regularizers import l2
from keras.utils import plot_model


def MCNN_model(img_rows, img_cols, channel=3, num_class=None):
    # Distortion Type Classifier
    input = Input(shape=(img_rows, img_cols, channel))
    conv1_7x7= Convolution2D(50,7,7,name='conv1_7x7_50',activation='relu')(input)#,W_regularizer=l2(0.0002)
    pool2_2x2= MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode='valid',name='pool2')(conv1_7x7)
    poll_flat = Flatten()(pool2_2x2)
    #MLP
    fc_1 = Dense(200,name='fc_1',activation='relu')(poll_flat)
    drop_fc = Dropout(0.5)(fc_1)
    out = Dense(5,name='fc_2',activation='sigmoid')(drop_fc)
    
    #IQA 1 ie for jpeg
    conv1= Convolution2D(50,7,7,name='conv1_jpeg',activation='relu')(input)#,W_regularizer=l2(0.0002)
    pool2= MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode='valid',name='pool2_jpeg')(conv1)
    poll = Flatten()(pool2)
    #MLP
    fc_11 = Dense(200,name='fc_1_jpeg',activation='relu')(poll)
    drop_fc11 = Dropout(0.5)(fc_11)
    out_1 = Dense(5,name='fc_2_jpeg',activation='sigmoid')(drop_fc11)
    
    #combine models     
    out_2 = keras.layers.Multiply()([out,out_1])
    #out_2 = keras.layers.Average()(out_2)
    out_2 = keras.layers.Dense(5)(out_2)
    
    # Create model  
    model = Model(input=input, output=out_2)
    # Load cnn pre-trained data 
    #model.load_weights('models/weights.h5')#NOTE 
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
    model.compile(optimizer=adam, loss='mean_absolute_error', metrics=['accuracy'])  
    return model 


model = MCNN_model(16, 16, 3)
plot_model(model, to_file='model_1.png', show_shapes=True)

