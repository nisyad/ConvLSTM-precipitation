# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 21:05:21 2019

@author: sumedh
"""
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.optimizers import Adam


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
                       input_shape = ip_shape,                       #activation = act,
                       padding = 'same',
                       data_format = 'channels_first',
                       return_sequences = True),
            ConvLSTM2D(filters = f2,
                       kernel_size = k_size,                       #activation = act,
                       padding = 'same',
                       data_format = 'channels_first',
                       return_sequences = False),
            Conv2D(filters = 1,
                   kernel_size = (1, 1),
                   activation = 'relu',
                   padding = 'same',
                   data_format = 'channels_first')])
            
    adam = Adam(lr = lr, decay = dec)
    model.compile(loss = loss, optimizer = adam, metrics=['accuracy'])
    
    return model


params = {'lr': (0.5, 5, 10),
          'batch_size': (2, 30, 10),
          'epochs': [150],
          'dropout': (0, 0.5, 5),
          'optimizer': [Adam, Nadam, RMSprop],
          'losses': [logcosh, binary_crossentropy],
          'activation':[relu, elu],
          'last_activation': [sigmoid]}