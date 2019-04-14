# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 17:29:14 2019

@author: sumedh
"""

import xarray as xr
import numpy as np
from datetime import datetime
import pandas as pd 

def create_data(filename, date = [], lat , lon, temp_data = ''):
    '''
    Creates the data required for the Conv LSTM script
    input :
        filename: name of the netcdf file to pick up precipitation data from
        date : a list/tuple containing the start and end dates
        lat : a list/tuple containing the min and max latitude
        lon : a list/tuple containing the min and max longitude
        temp_data: if need tempearture data a third dimension, provide file path
    
    output :
        A numpy array with the precipitaion data and timestamp/temperature features
        The numpy array is saved to the local directory.
    '''
    
    if len(lat) != 2 or len(lon) != 2:
        print('Latitude and Longitude need to be a tuple')
        break
    
    dat = xr.open_dataset(filename)
    
    if not date:
        date_array = np.array(dat.time)
        d1 = pd.to_datetime(str(np.min(date_array)))
        d2 = pd.to_datetime(str(np.max(date_array)))
        date = [d1.strftime('%Y-%m-%d'), d2.strftime('%Y-%m-%d')]
    
    sliced = dat.precip.sel(time = slice(date[0],date[1]),
                            lat = slice(lat[0],lat[1]),
                            lon = slice(lon[0],lon[1]))
    sliced = np.array(sliced)
    
    d = d1.strftime('%Y-%m-%d').split('-')
    base = datetime(int(d[0]), int(d[1]), int(d[2]))
    date_list = [base + datetime.timedelta(days=x) for x in range(0, sliced.shape[0])]
    t = np.zeros((sliced.shape[0],sliced.shape[1],sliced.shape[2]))
    
    for i in range(0,sliced.shape[0]):
        t[i].fill(date_list[i].month)
    
    sliced = np.stack((sliced,t), axis = 3)
    
    if temp_data:
        temp_slice = np.load(temp_data)
        sliced = np.stack((sliced,temp_slice), axis = 3)
    
    np.save('sliced',sliced)
        
    return sliced
