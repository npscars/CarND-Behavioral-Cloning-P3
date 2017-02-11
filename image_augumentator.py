# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 19:50:07 2017

@author: SHAHN
"""
import numpy as np
#from skimage.transform import resize
#import matplotlib.pyplot as plt
import scipy.ndimage as snd
import cv2

def gather_image_information(dataDir,percentZeroAngle=5,addHighAngle=True,highAngle=0.1,percentValData = 10):
    #header   = np.loadtxt(dataDir+'driving_log.csv',dtype=str,delimiter=',',)[0]
    img_name = np.loadtxt(dataDir+'driving_log.csv',dtype=str,delimiter=',',skiprows = 1,usecols=range(0,3))
    img_data = np.loadtxt(dataDir+'driving_log.csv',dtype = float,delimiter=',',skiprows=1,usecols=range(3,7))

    # remove random 20% out of all 0 deg steering angle images
    #idxNonZero = np.transpose(np.asarray(np.nonzero(img_data[:,0])))
    idxZero = np.transpose(np.asarray(np.where(img_data[:,0]==0)))
    idxZero = np.random.choice(idxZero[:,0],np.int(np.round((percentZeroAngle/100)*len(idxZero))),replace=False) # only 5% zero angle data
    idxNonZero = np.transpose(np.asarray(np.where(np.logical_or(img_data[:,0]>0,img_data[:,0]<0))))
    idxConsiderForModel = np.concatenate((idxZero,idxNonZero[:,0]),axis=0)

    # further addition of same high steering angle data based on results of a bit of run
    if addHighAngle == True:
        idxHighSteer = np.transpose(np.asarray(np.where(np.logical_or(img_data[:,0]>highAngle,img_data[:,0]<-highAngle))))
        idxConsiderForModel = np.concatenate((idxConsiderForModel,idxHighSteer[:,0]),axis=0)

    np.random.shuffle(idxConsiderForModel)

    firstValDataidx = np.int(len(idxConsiderForModel) - len(idxConsiderForModel)*(percentValData/100)) # take last 20% data as validation after full shuffle
    val_name = img_name[idxConsiderForModel[firstValDataidx:],:]
    val_data = img_data[idxConsiderForModel[firstValDataidx:],:]

    img_name = img_name[idxConsiderForModel[0:firstValDataidx],:]
    img_data = img_data[idxConsiderForModel[0:firstValDataidx],:]
    
    return img_name,img_data,val_name,val_data

def load_image_as_matrix(xdata,ydata,dataDir,i_lcr, offsetSteer = 0.15):
    #select left , right or centre randomly
    X_data = snd.imread(dataDir + xdata[i_lcr][2:-1])
    # augumentation of data
    if i_lcr==1: #left
        Y_data = np.min((ydata + offsetSteer,1)) # no more than 1
    elif i_lcr==2: #right
        Y_data = np.max((ydata - offsetSteer,-1)) # no less than -1
    else:
        Y_data = ydata
    return X_data,Y_data

def flip_camera_images(image,ydata):
    ind_flip = np.random.randint(2)
    if ind_flip==0:
        image1  = cv2.flip(image,1) #vertical flip
        ydata = ydata * -1.0
    else:
        image1 = image
        ydata  = ydata
    return image1, ydata
    
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def add_random_shadow(image):
    top_y = 320*np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    #random_bright = .25+.7*np.random.uniform()
    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image
