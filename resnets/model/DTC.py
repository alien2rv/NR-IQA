# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:34:19 2019

@author: rravela

"""
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Convolution2D, Input,Activation, ZeroPadding2D, MaxPooling2D, Flatten, merge
from keras.optimizers import SGD, Adam
from keras.objectives import sparse_categorical_crossentropy as scc
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
from keras.utils import plot_model

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

def image_split_d(k,blk_size,label):
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

def dtc_resnet(weights_path = None):
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
    if weights_path:
        model.load_weights(weights_path)
    
    #sgd = SGD(decay=0., lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
    return model