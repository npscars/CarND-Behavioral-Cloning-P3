# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D
#from keras.optimizers import Adam
import numpy as np
#import os
from skimage.transform import resize
import matplotlib.pyplot as plt
import scipy.ndimage as snd
import cv2
#from sklearn.model_selection import train_test_split
#from sklearn.utils import shuffle
# fix random seed for reproducibility
# seed = 7
# numpy.random.seed(seed)
# load dataset

dataDir  = 'D:/python/SDC/Projects/P3/sampleDataTrack1/data/'
header   = np.loadtxt(dataDir+'driving_log.csv',dtype=str,delimiter=',',)[0]
img_name = np.loadtxt(dataDir+'driving_log.csv',dtype=str,delimiter=',',skiprows = 1,usecols=range(0,3))
img_data = np.loadtxt(dataDir+'driving_log.csv',dtype = float,delimiter=',',skiprows=1,usecols=range(3,7))
# remove random 20% out of all 0 deg steering angle images
#idxNonZero = np.transpose(np.asarray(np.nonzero(img_data[:,0])))
idxZero = np.transpose(np.asarray(np.where(img_data[:,0]==0)))
idxZero = np.random.choice(idxZero[:,0],np.int(np.round(.05*len(idxZero))),replace=False) # only 30% zero angle data
idxNonZero = np.transpose(np.asarray(np.where(np.logical_or(img_data[:,0]>0,img_data[:,0]<0))))
idxConsiderForModel = np.concatenate((idxZero,idxNonZero[:,0]),axis=0)

# further addition of same high steering angle data based on results of a bit of run
idxHighSteer = np.transpose(np.asarray(np.where(np.logical_or(img_data[:,0]>0.1,img_data[:,0]<-0.1))))
idxConsiderForModel = np.concatenate((idxConsiderForModel,idxHighSteer[:,0]),axis=0)

np.random.shuffle(idxConsiderForModel)

#firstValDataidx = np.int(len(idxConsiderForModel) - len(idxConsiderForModel)*0.2) # take last 20% data as validation after full shuffle
#val_name = img_name[idxConsiderForModel[firstValDataidx:],:]
#val_data = img_data[idxConsiderForModel[firstValDataidx:],:]
firstValDataidx = len(idxConsiderForModel)+1
img_name = img_name[idxConsiderForModel[0:firstValDataidx],:]
img_data = img_data[idxConsiderForModel[0:firstValDataidx],:]

def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def generate_train_data_batch(xdata,ydata,batch_size = 100,dataDir  = 'D:/python/SDC/Projects/P3/sampleDataTrack1/data/'):
    # normalize/resize/change RGB to YUV as per nVidia paper
    normMin, normMax = -0.5, 0.5 # normalize and combine image between -0.5 to 0.5 in all 3 RGB or YUV
    offsetSteer      = 0.25
    X_data = np.empty((batch_size,66,200,3))
    Y_data = np.empty((batch_size,1))   
    while 1:
        for i_batch in range(batch_size):
            i_select = np.random.randint(len(xdata))
            i_lcr    = 0 #np.random.randint(3) #select left , right or centre randomly
            temp = snd.imread(dataDir + xdata[i_select,i_lcr][2:-1])
            
            # augumentation of data
            if i_lcr==1:
                ydata[i_select,0] = ydata[i_select,0] + offsetSteer
            elif i_lcr==2:
                ydata[i_select,0] = ydata[i_select,0] - offsetSteer
            else:
                ydata[i_select,0] = ydata[i_select,0]
            
            # change brightness
            temp = augment_brightness_camera_images(temp)
            
            # flip images
            ind_flip = np.random.randint(2)
            if ind_flip==0:
                temp = cv2.flip(temp,1) #vertical flip
                Y_data[i_batch,:] = -ydata[i_select,0]
            else:
                Y_data[i_batch,:] = ydata[i_select,0]
            
            # part of input to model fitting so don't change below
            temp = cv2.cvtColor(temp, cv2.COLOR_RGB2YUV)
            temp = resize(temp[50:-10,:,:],(66,200)) # already normalizes to between 0-1 from 0-255
            X_data[i_batch,:] = np.reshape(normMin + ((temp-0)*(normMax-normMin))/(1-0),(1,66,200,3))
           # Y_data[i_batch,:] = ydata[i_select,0]
        yield X_data,Y_data

# split into input (X) and output (Y) variables
#X = X_data
#Y_data = img_data[:,0]
#X_train, X_val, y_train, y_val = train_test_split(X_data, Y_data, test_size=0.2, random_state=0)
# create model from http://machinelearningmastery.com/save-load-keras-deep-learning-models/

model = Sequential()
model.add(Convolution2D(24, 5, 5, input_shape=(66, 200, 3),activation = 'relu',init = 'uniform'))
model.add(Convolution2D(36, 5, 5,activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(48, 5, 5,activation = 'relu'))
model.add(Convolution2D(64, 3, 3,activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, 3, 3,activation = 'relu'))
model.add(Flatten())
model.add(Dense(100,init='uniform', activation='relu'))
model.add(Dropout(p=0.5))
model.add(Dense(50,init='uniform', activation='relu'))
model.add(Dropout(p=0.5))
model.add(Dense(10,init='uniform', activation='relu'))
model.add(Dropout(p=0.5))
model.add(Dense(1, name='y_pred'))
#load saved weights
#model.load_weights('model.h5')
# Compile model
model.compile(loss='mse', optimizer='adam')
# Fit the model
hist = model.fit_generator(generate_train_data_batch(img_name,img_data), nb_epoch=5, samples_per_epoch=3000)
print(hist.history)
# evaluate the model
#scores = model.evaluate_generator(generate_train_data_batch(val_name,val_data),val_samples = 500)
#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
 
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
# later...
'''
# load json and create model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
'''
# evaluate loaded model on test data
#loaded_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)

''' def normalize_grayscale(image_data):\n",
    "    a = -0.5\n",
    "    b = 0.5\n",
    "    grayscale_min = 0\n",
    "    grayscale_max = 255\n",
    "    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )
'''

