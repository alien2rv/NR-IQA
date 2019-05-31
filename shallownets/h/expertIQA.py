from keras.layers import Input,InputLayer, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Sequential,Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,LearningRateScheduler
import keras
from keras import backend as K

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
	
def gblur_model(weights_path=None):
    nb_filters = 50
    nb_conv = 7

    model = Sequential()
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv), input_shape=(32, 32,3), activation = 'relu', name='conv1') )
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf", name='pool_1'))
    
    model.add(Flatten( name='flatten_1'))
    
    model.add(Dense(units = 800, activation = 'relu',name='FC_1'))
    model.add(Dropout(0.1,name='dropout_1'))

    model.add(Dense(units = 800, activation = 'relu',name='FC_2'))
    model.add(Dropout(0.1, name='dropout_2'))

    model.add(Dense(1, name='fc_3'))
    model.add(Activation('linear'))
    model.summary()
    
#    if weights_path:
 #       model.load_weights(weights_path)
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00005, amsgrad=False))
    return model

def awgn_model(weights_path=None):
    nb_filters = 50
    nb_conv = 7

    model = Sequential()
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv), input_shape=(32, 32,3), activation = 'relu', name='conv1') )
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf", name='pool_1'))
    
    model.add(Flatten( name='flatten_1'))
    
    model.add(Dense(units = 800, activation = 'relu',name='FC_1'))
    model.add(Dropout(0.1,name='dropout_1'))

    model.add(Dense(units = 800, activation = 'relu',name='FC_2'))
    model.add(Dropout(0.1, name='dropout_2'))

    model.add(Dense(1, name='fc_3'))
    model.add(Activation('linear'))
    model.summary()
    
#    if weights_path:
 #       model.load_weights(weights_path)
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00005, amsgrad=False))
    return model
	
def jpeg_model(weights_path=None):
    nb_filters = 50
    nb_conv = 7

    model = Sequential()
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv), input_shape=(32, 32,3), activation = 'tanh', name='conv1') )
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf", name='pool_1'))
    
    model.add(Flatten( name='flatten_1'))
    
    model.add(Dense(units = 800, activation = 'relu',name='FC_1'))
    model.add(Dropout(0.1,name='dropout_1'))

    model.add(Dense(units = 800, activation = 'relu',name='FC_2'))
    model.add(Dropout(0.1, name='dropout_2'))

    model.add(Dense(1, name='fc_3'))
    model.add(Activation('sigmoid'))
    model.summary()
    
#    if weights_path:
 #       model.load_weights(weights_path)
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00005, amsgrad=False))
    return model
	
def jp2k_model(weights_path=None):
    nb_filters = 50
    nb_conv = 7

    model = Sequential()
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv), input_shape=(32, 32,3), activation = 'tanh', name='conv1') )
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf", name='pool_1'))
    
    model.add(Flatten( name='flatten_1'))
    
    model.add(Dense(units = 800, activation = 'relu',name='FC_1'))
    model.add(Dropout(0.1,name='dropout_1'))

    model.add(Dense(units = 800, activation = 'relu',name='FC_2'))
    model.add(Dropout(0.1, name='dropout_2'))

    model.add(Dense(1, name='fc_3'))
    model.add(Activation('sigmoid'))
    model.summary()
    
#    if weights_path:
 #       model.load_weights(weights_path)
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00005, amsgrad=False))
    return model