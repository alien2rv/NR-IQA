# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:42:54 2019

@author: rravela
"""
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Convolution2D, Input,Activation, ZeroPadding2D, MaxPooling2D, Flatten, merge
from keras.optimizers import SGD, Adam
from keras.objectives import sparse_categorical_crossentropy as scc
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from keras.utils import plot_model
def awgn_resnet():
    # In order to make things less confusing, all layers have been declared first, and then used
    
    # declaration of layers
    input_img = Input((32, 32,3), name='input_layer')
    zeroPad1 = ZeroPadding2D((1,1), name='zeroPad1', dim_ordering='tf')
    zeroPad1_2 = ZeroPadding2D((1,1), name='zeroPad1_2', dim_ordering='tf')
    layer1 = Convolution2D(64, 3, 3, subsample=(2, 2), init='he_uniform', name='major_conv', dim_ordering='tf')
    layer1_2 = Convolution2D(128, 3, 3, subsample=(2, 2), init='he_uniform', name='major_conv2', dim_ordering='tf')
    zeroPad2 = ZeroPadding2D((1,1), name='zeroPad2', dim_ordering='tf')
    zeroPad2_2 = ZeroPadding2D((1,1), name='zeroPad2_2', dim_ordering='tf')
    layer2 = Convolution2D(64, 3, 3, subsample=(1,1), init='he_uniform', name='l1_conv', dim_ordering='tf')
    layer2_2 = Convolution2D(128, 3, 3, subsample=(1,1), init='he_uniform', name='l1_conv2', dim_ordering='tf')


    zeroPad3 = ZeroPadding2D((1,1), name='zeroPad3', dim_ordering='tf')
    zeroPad3_2 = ZeroPadding2D((1,1), name='zeroPad3_2', dim_ordering='tf')
    layer3 = Convolution2D(64, 3, 3, subsample=(1, 1), init='he_uniform', name='l2_conv', dim_ordering='tf')
    layer3_2 = Convolution2D(128, 3, 3, subsample=(1, 1), init='he_uniform', name='l2_conv2', dim_ordering='tf')

    layer4 = Dense(512, activation='relu', init='he_uniform', name='dense1')
    layer5 = Dense(256, activation='relu', init='he_uniform', name='dense2')
    
    final = Dense(1, activation='linear', init='he_uniform', name='regressor')
    
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
    
    
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-4, amsgrad=False)
    model.compile(loss='mean_squared_error', optimizer= adam)
    return model