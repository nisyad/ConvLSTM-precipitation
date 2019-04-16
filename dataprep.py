# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 00:27:31 2019

@author: sumedh
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def prep_data(data, validation_split =0.3, window_size = 20, scale = False):
    '''
    Split data into training and validation splits
    '''
    dat = {'train_x': np.empty((0,0)),
            'train_y': np.empty((0,0)),
            'val_x': np.empty((0,0)),
            'val_y': np.empty((0,0)),
            'scaler_train':[],
            'scaler_val':[]}
    
    n = int(len(data)*(1- validation_split))
    train = data[:n]
    val = data[n:]
    
    if scale:
        scaler = MinMaxScaler(feature_range = (-1,1))
        scaler_train = scaler.fit(train[:,:,:,0].reshape(-1,1))
        scale_t = scaler_train.transform(train[:,:,:,0].reshape(-1,1)).reshape(train.shape[:3])
        dat['scaler_train'].append(scaler_train)
        train[:,:,:,0] = scale_t
        scaler_val = scaler.fit(val[:,:,:,0].reshape(-1,1))
        scale_v = scaler_val.transform(val[:,:,:,0].reshape(-1,1)).reshape(val.shape[:3])
        val[:,:,:,0] = scale_v
        dat['scaler_val'].append(scaler_val)
        
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
    
    dat['train_x'] = x
    dat['train_y'] = y[:,-1]
    dat['val_x'] = x_val
    dat['val_y'] = y_val[:,-1]
    
    return dat 

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