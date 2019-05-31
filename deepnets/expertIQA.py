from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam
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
	
def blur_VGG(weights_path=None):
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
    
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00005, amsgrad=False))
    return model
	
def awgn_VGG(weights_path=None):
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
    
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00005, amsgrad=False))
    return model
	
def jpeg_VGG(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(32,32,3)))
    model.add(Conv2D(32,kernel_size=(3, 3), activation='tanh'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32,kernel_size=(3, 3), activation='tanh'))
    model.add(MaxPooling2D((2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64,kernel_size=(3, 3), activation='tanh'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64,kernel_size=(3, 3), activation='tanh'))
    model.add(MaxPooling2D((2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128,kernel_size=(3, 3), activation='tanh'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128,kernel_size=(3, 3), activation='tanh'))
    model.add(MaxPooling2D((2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256,kernel_size=(3, 3), activation='tanh'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256,kernel_size=(3, 3), activation='tanh'))
    model.add(MaxPooling2D((2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,kernel_size=(3, 3), activation='tanh'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,kernel_size=(3, 3), activation='tanh'))
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(units = 512, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    if weights_path:
        model.load_weights(weights_path)
    
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00005, amsgrad=False))
    return model
	
def jp2k_VGG(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(32,32,3)))
    model.add(Conv2D(32,kernel_size=(3, 3), activation='tanh'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(32,kernel_size=(3, 3), activation='tanh'))
    model.add(MaxPooling2D((2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64,kernel_size=(3, 3), activation='tanh'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64,kernel_size=(3, 3), activation='tanh'))
    model.add(MaxPooling2D((2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128,kernel_size=(3, 3), activation='tanh'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128,kernel_size=(3, 3), activation='tanh'))
    model.add(MaxPooling2D((2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256,kernel_size=(3, 3), activation='tanh'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256,kernel_size=(3, 3), activation='tanh'))
    model.add(MaxPooling2D((2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,kernel_size=(3, 3), activation='tanh'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,kernel_size=(3, 3), activation='tanh'))
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(units = 512, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    if weights_path:
        model.load_weights(weights_path)
    
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00005, amsgrad=False))
    return model