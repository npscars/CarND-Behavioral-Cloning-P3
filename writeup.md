#**Behavioral Cloning** 

---

**Behavrioal Cloning Project**

[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train and validate the model
* drive.py for driving the car in autonomous mode (modified)
* model.h5 containing a trained convolution neural network 
* image_augumentator.py contains helper functions to run model.py, loads and augments images
* writeup.md summarizing the results (this file :))

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing. I have also included runJ1.mp4 which includes the autonomous run of track 1 for your reference.
```sh
python drive.py model.json
```

####3. Submssion code is usable and readable

The model.py file contains the python generators for training and validation data and also the NVIDIA inspired KERAS convolution neural network model. It contains comments to explain how the code works. I also included well documented image_augumentator.py as it includes helper functions for generating and loading images and also functions for augumentating the images. 

I am indebted on general help from articles retrived using Google and CarND confluence forum. Also help from Vivek Yadav's articles such as [chatbot article](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.k63595b8o) and [github code](https://github.com/vxy10/ImageAugmentation) 

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a 5 Convolution Neural Network (CNN). First three layers were with 5x5 filter sizes and 2x2 strides and depths between 24 and 48 (model.py lines 59-63). Last two CNNs were with 3x3 filter sizes and depth of 64.

Like the [NVIDIA pipeline](https://arxiv.org/pdf/1604.07316.pdf), three fully connected layer were used, after the 5 feature extractor CNN layers, to act as a controller as potentially suggested in the paper. 

The model includes RELU layers to introduce nonlinearity and also TANH activation layer for the last Fully Connected (FC) layer (10 neurons) to have output from -1 to 1 and not only 0 to 1 as in RELU. The data is normalized in the training and validation data generator. 

####2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 79-80). There were couple of problems in my code which were realised due to increase of loss with every epoch. As soon as i solved those problems, the loss started decreasing with every epoch. After around 6-7 epoch the loss was not decreasing further and hence I selected 7 as my number of epoch for final fitting of model. 

 I added dropout layers to the model in order to reduce overfitting. I had found out from my last project that dropouts were potentially more effective in FC layers as compared to CNN layers and hence I added them in all FC layers only.

The model was tested by running it through the simulator and ensuring that the vehicle stayed on the track 1. Video of it is also included as [runJ1.mp4](./runJ1.mp4) 

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually but initial value selected was 0.0001 instead of default 0.001 (model.py line 75).

####4. Appropriate training data

First I started using sample Udacity data. It contained lots of zero angle data and I decided to use only 5% of zero angle data otherwise my model would predict lots of zero angle and ignore other turning data. Then I added more turning data or centering of car in lane i.e. >0.1 & <-0.1 angle data. This is what is done in "gather_image_information" function of image_augumentator.py file.

This function also includes augumentation of the images like adding shadow, flipping images, adding brightness, and using left/right camera images in addition to centre image.

I trained the model but as soon as i run the model, the car was moving out of center lane. So clearly it needed more data in terms of recovery data. I would like to thank Rohit Patil (my mentor) for additional recovery data. It would have been difficult to complete this project without his help.

Finally after adding all together i.e. model with dropout, image with augumentation, and additional recovery data, the car was able to ride on track 1 without any issues.

The cars runs on track 2 as well but just crashes after the last right turn after down hill slope. I think more training data is required and hopefully when i get more time I will work on it.

###Model Architecture and Training Strategy

####1. Solution Design Approach
I describe the model architecture in section "An appropriate model arcthiecture has been employed" above.

At the end of the process, the vehicle is able to drive autonomously around the track 1 without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24).