# **Behavioral Cloning** 

**Behavioral Cloning Project**

The objective of the project is to train a neural network to mimic the behaviour of a driver by reading the images generated from the simulator


#### Files 
* Behavioral Model.ipynb - Jupyter notebook
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 

#### How to run the file
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5 run4  
```

Video from snapshot of images can be created using the code

```
python video.py run4
```

### Model Architecture and Training Strategy

> The model is modified version of nVidia self driving car model. Lambda function is applied to normalize the image size, followed by cropping out the irrelevant parts of the road images. The kernel size and strides are kept as per the nvidia paper (three CNN layers with strides and two without strides). Drop out is added between the layers to reduce the number of parameters to be learned during a feedforward process. This also helps in reducing overfitting. The layers are then connected to dense layers - 100,50,10,1 to estimate the steering angle

_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
dropout_1 (Dropout)          (None, 5, 37, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
dropout_2 (Dropout)          (None, 1, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300    
_________________________________________________________________
activation_1 (Activation)    (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
activation_2 (Activation)    (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
activation_3 (Activation)    (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
__________________________

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### Model parameter tuning

The model used rmprop optimizer, so the learning rate was not tuned manually. To avoid gradient explosion, the gradients are clipped to a value of 100. The batch size is kept at 32.

#### Data Augumentation
1. Image Flip
2. Used images from left and right camera

#### Data Collection Process
> The car is driven for close to an hour by navigation with the help of a mouse. This help in understanding when the data is good enough to drive on a straight road. For turns, the car is driven with the help of keyboard to provide quick and large enough steering angles. For the bridge section, the car is kept close to a straight line. By providing a correction factor for left and right images, the car is able to learn how to cross the bridge even if is near the walls. Two laps with frequent pauses were covered for recovering from left and right lanes. One additional round was covered by navigating the car with keyboard strokes6


