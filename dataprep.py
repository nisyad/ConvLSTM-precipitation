# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 00:27:31 2019

@author: sumedh
"""

import numpy as np

def prep_data(data, validation_split =0.3, window_size = 20):
    '''
    Split data into training and validation splits
    '''
    n = int(len(data)*(1- validation_split))
    train = data[:n]
    val = data[n:]
    
    x = train[:-1]
    y = train[1:,:,:,0]
    x = reshape_data(x)
    y = reshape_data(y,True)
    x = make_windows(x, window_size)
    y = make_windows(y, window_size, True)
    
    x_val = val[:-1]
    y_val = val[1:,:,:,0]
    x_val = reshape_data(x_val)
    y_val = reshape_data(y_val, True)
    x_val = make_windows(x_val, window_size)
    y_val = make_windows(y_val, window_size, True)
    
    return x , y[:,-1], x_val, y_val[:,-1]


def reshape_data(d, b = False):
    '''
    reshape data into proper dimensions
    '''
    channels = 3
    if b:
        channels = 1
        
    return d.reshape(d.shape[0], channels, d.shape[1], d.shape[2])

def make_windows(d, size, b = False):
    '''
    make sliding windows
    '''
    channels = 3
    if b:
        channels = 1
    w = d.shape[0]-size+1    
    t = np.zeros((w, size, channels, d.shape[2], d.shape[3]))
    for i in range(w):
        t[i] = d[i:i+size]
    
    return t