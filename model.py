# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:05:21 2019

@author: sumedh
"""
import os
import dataprep
import numpy as np
from glob import glob
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout


def create_model(ip_shape, k_size, lr= 0.001 , dec = 0.0, f1 = 16, f2 = 8, loss = 'mse'):
    '''
    Creates the convolutional model 
    input:
        ip_shape : The input shape of the features (20,3,32,32)
        k_size : size  of the kernel (3,3)
        lr : learning rate (0-1)
        dec: decay rate
        f1: filters in the first convolutional layer
        f2: fitlers in the second convlutional layer
        loss : the loss function to consider
        
    output:
        The model trained by the given parameters
    '''
    model = Sequential([
            ConvLSTM2D(filters = f1,
                       kernel_size = k_size, 
                       input_shape = ip_shape,
                       activation = 'tanh',
                       padding = 'same',
                       data_format = 'channels_first',
                       return_sequences = True),
            BatchNormalization(),
            Dropout(0.5),
            ConvLSTM2D(filters = f2,
                       kernel_size = k_size,     
                       activation = 'tanh',
                       padding = 'same',
                       data_format = 'channels_first',
                       return_sequences = False),
            BatchNormalization(),
            Dropout(0.5),
            Conv2D(filters = 1,
                   kernel_size = (1, 1),
                   activation = 'relu',
                   padding = 'same',
                   data_format = 'channels_first'),
            BatchNormalization(),
            Dropout(0.5)])
            
    adam = Adam(lr = lr, decay = dec)
    model.compile(loss = loss, optimizer = adam, metrics=['accuracy'])
    
    return model

path = os.getcwd()
data = np.load(glob(path+'\\data\\*')[0])
x_train, y_train, x_val, y_val = dataprep.prep_data(data = data, validation_split = 0.3, window_size = 10)
model = create_model(x_train.shape[1:], (3,3), 0.005, 0.0, 32, 32, 'mse')
history = model.fit(x_train, y_train, epochs = 50, validation_data = (x_val, y_val))

