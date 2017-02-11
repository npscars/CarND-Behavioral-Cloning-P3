# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import cv2
import image_augumentator as ia

# load dataset
#dataDir  = 'data/'
dataDir = 'D:/python/SDC/Projects/P3/Data/sampleDataTrack1/data/'
img_name,img_data,val_name,val_data = ia.gather_image_information(dataDir,percentZeroAngle = 5, addHighAngle = True,percentValData = 10)

def generate_val_data(xdata,ydata,dataDir,batch_size = 64):
    normMin, normMax = -0.5, 0.5 # normalize and combine image between -0.5 to 0.5 in all 3 RGB or YUV
    X_data = np.empty((batch_size,66,200,3))
    Y_data = np.empty((batch_size,1))
    while 1:
        for i_batch in range(batch_size):
            i_select = np.random.randint(len(xdata))
            i_lcr    = 0 # as in autonomous mode only centre image is avaialble
            temp, angle = ia.load_image_as_matrix(xdata[i_select,:],ydata[i_select,0],dataDir,i_lcr) 
            temp = cv2.resize(temp[55:-20,:,:],(200,66),interpolation = cv2.INTER_AREA) # already normalizes to between 0-1 from 0-255
            temp = cv2.cvtColor(temp, cv2.COLOR_RGB2YUV)
            X_data[i_batch,:] = np.reshape(normMin + ((temp-0)*(normMax-normMin))/(1-0),(1,66,200,3))
            Y_data[i_batch,0] = angle
        yield X_data,Y_data

def generate_train_data_batch(xdata,ydata,dataDir,batch_size = 64):
    # normalize/resize/change RGB to YUV as per nVidia paper
    normMin, normMax = -0.5, 0.5 # normalize and combine image between -0.5 to 0.5 in all 3 RGB or YUV
    X_data = np.empty((batch_size,66,200,3))
    Y_data = np.empty((batch_size,1))   
    while 1:
        for i_batch in range(batch_size):
            i_select = np.random.randint(len(xdata))
            i_lcr    = np.random.randint(3) #select left , right or centre randomly
            temp, angle = ia.load_image_as_matrix(xdata[i_select,:],ydata[i_select,0],dataDir,i_lcr, offsetSteer = 0.15)                          
            # change brightness
            temp = ia.augment_brightness_camera_images(temp)
            # add shadow
            temp = ia.add_random_shadow(temp)
            # flip randomly
            temp, angle = ia.flip_camera_images(temp,angle)
            
            # part of input to model fitting so don't change below
            temp = cv2.resize(temp[55:-20,:,:],(200,66),interpolation = cv2.INTER_AREA) # already normalizes to between 0-1 from 0-255
            temp = cv2.cvtColor(temp, cv2.COLOR_RGB2YUV)
            X_data[i_batch,:] = np.reshape(normMin + ((temp-0)*(normMax-normMin))/(1-0),(1,66,200,3))
            Y_data[i_batch,:] = angle
        yield X_data,Y_data

model = Sequential()
model.add(Convolution2D(24, 5, 5, input_shape=(66, 200, 3), subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(36, 5, 5,subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(48, 5, 5,subsample = (2,2), activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Convolution2D(64, 3, 3, activation = 'relu'))
model.add(Flatten())
model.add(Dense(100,init='uniform', activation='relu'))
model.add(Dropout(p=0.5))
model.add(Dense(50,init='uniform', activation='relu'))
model.add(Dropout(p=0.5))
model.add(Dense(10,init='uniform', activation='tanh'))
model.add(Dropout(p=0.5))
model.add(Dense(1,name='y_pred'))
#load saved weights
model.load_weights('model.h5')
# Compile model
opt = Adam(lr=0.0001)
model.compile(loss='mse', optimizer=opt)
# Fit the model
hist = model.fit_generator(generate_train_data_batch(img_name,img_data,dataDir), nb_epoch=5, samples_per_epoch=20480, \
                           validation_data = generate_val_data(val_name,val_data,dataDir), nb_val_samples=64)
 
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
