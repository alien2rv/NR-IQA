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

dataset = pd.read_csv('D:\\Databases\\lcn_data\\live\\labels\\DTC_live.csv')
b = dataset.iloc[:].values
label = np.zeros((len(b),len(b[0])+1))
label[:,:-1] = b

label_train = np.concatenate((label[:120],label[175:293],label[344:444],label[489:589]))
label_test = np.concatenate((label[120:151],label[293:321],label[444:469],label[589:614]))
label_valid = np.concatenate((label[151:175],label[321:344],label[469:489],label[614:634]))

image =np.concatenate((jpeg_lcn,jpeg2000_lcn,blur_lcn,awgn_lcn))
image_train = np.concatenate((image[:120],image[175:293],image[344:444],image[489:589]))
image_test = np.concatenate((image[120:151],image[293:321],image[444:469],image[589:614]))
image_valid = np.concatenate((image[151:175],image[321:344],image[469:489],image[614:634]))
image = []


#training the model
#distortion type classifier
X_train, Y_train = image_split_d(image_train,32,label_train)
X_train = np.array(X_train)
Y_train = np.array(Y_train)

X_test,Y_test = image_split_d(image_test,32,label_test)
X_test = np.array(X_test)
Y_test = np.array(Y_test)

X_valid,Y_valid = image_split_d(image_valid,32,label_valid)
X_valid = np.array(X_valid)
Y_valid = np.array(Y_valid)

res = dtc_model()
res.summary()
plot_model(res, to_file='D:\\thesis\\coding\\shallownets\\dtc.png', show_shapes = True)
res.fit(X_train, Y_train[:,:4], validation_data=(X_valid, Y_valid[:,:4]),
                          epochs=20,verbose=1)
pred_dtc = res.predict(X_test)
res.save_weights("D:\\thesis\\coding\\shallownets\\DTC.h5")
np.save("D:\\thesis\\coding\\shallownets\\DTC.npy",pred_dtc,  allow_pickle=True, fix_imports=True)


# gblur
dataset = pd.read_csv('D:\\Databases\\lcn_data\\live\\labels\\live_gblur_sorted.csv')
b = dataset.iloc[:].values
label = np.zeros((len(b),3))
label[:,:-1] = b

image_train = blur_lcn[:95]
image_valid = blur_lcn[95:120]
label_train = label[:95]
label_valid = label[95:120]

X_train, Y_train = image_split(image_train,32,label_train)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_valid,Y_valid = image_split(image_valid,32,label_valid)
X_valid = np.array(X_valid)
Y_valid = np.array(Y_valid)

res = gblur_model()
res.summary()
plot_model(res, to_file='D:\\thesis\\coding\\shallownets\\gblur.png', show_shapes = True)
res.fit(X_train, (Y_train[:,:1]*0.01), validation_data=(X_valid, (Y_valid[:,:1]*0.01)),
                          epochs=20,verbose=1)
res.save_weights("D:\\thesis\\coding\\shallownets\\gblur.h5")

# awgn
dataset = pd.read_csv('D:\\Databases\\lcn_data\\live\\labels\\live_wn_sorted.csv')
b = dataset.iloc[:].values
label = np.zeros((len(b),3))
label[:,:-1] = b

image_train = awgn_lcn[:95]
image_valid = awgn_lcn[95:120]
label_train = label[:95]
label_valid = label[95:120]

X_train, Y_train = image_split(image_train,32,label_train)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_valid,Y_valid = image_split(image_valid,32,label_valid)
X_valid = np.array(X_valid)
Y_valid = np.array(Y_valid)

res = awgn_model()
res.summary()
plot_model(res, to_file='D:\\thesis\\coding\\shallownets\\awgn.png', show_shapes = True)
res.fit(X_train, (Y_train[:,:1]*0.01), validation_data=(X_valid, (Y_valid[:,:1]*0.01)),
                          epochs=20,verbose=1)
res.save_weights("D:\\thesis\\coding\\shallownets\\awgn.h5")

# jpeg
dataset = pd.read_csv('D:\\Databases\\lcn_data\\live\\labels\\live_jpeg_sorted.csv')
b = dataset.iloc[:].values
label = np.zeros((len(b),3))
label[:,:-1] = b

image_train = jpeg_lcn[:113]
image_valid = jpeg_lcn[113:144]
label_train = label[:113]
label_valid = label[113:144]

X_train, Y_train = image_split(image_train,32,label_train)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_valid,Y_valid = image_split(image_valid,32,label_valid)
X_valid = np.array(X_valid)
Y_valid = np.array(Y_valid)

res = jpeg_model()
res.summary()
plot_model(res, to_file='D:\\thesis\\coding\\shallownets\\jpeg.png', show_shapes = True)
res.fit(X_train, (Y_train[:,:1]*0.01), validation_data=(X_valid, (Y_valid[:,:1]*0.01)),
                          epochs=20,verbose=1)
res.save_weights("D:\\thesis\\coding\\shallownets\\jpeg.h5")

# jpeg
dataset = pd.read_csv('D:\\Databases\\lcn_data\\live\\labels\\live_jp2k_sorted.csv')
b = dataset.iloc[:].values
label = np.zeros((len(b),3))
label[:,:-1] = b

image_train = jpeg2000_lcn[:113]
image_valid = jpeg2000_lcn[113:141]
label_train = label[:113]
label_valid = label[113:141]

X_train, Y_train = image_split(image_train,32,label_train)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
X_valid,Y_valid = image_split(image_valid,32,label_valid)
X_valid = np.array(X_valid)
Y_valid = np.array(Y_valid)

res = jp2k_model()
res.summary()
plot_model(res, to_file='D:\\thesis\\coding\\shallownets\\jp2k.png', show_shapes = True)
res.fit(X_train, (Y_train[:,:1]*0.01), validation_data=(X_valid, (Y_valid[:,:1]*0.01)),
                          epochs=20,verbose=1)
res.save_weights("D:\\thesis\\coding\\shallownets\\jp2k.h5")