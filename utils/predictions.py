import tensorflow as tf
import tensorflow.keras.backend as K
import os, json, sys
import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.metrics import r2_score
from itertools import chain
sys.path.append("./models")
import pdb
from glob import glob
from scipy.signal import savgol_filter

"""
Function: r2_keras        
----------
Use: Loss function for neural networks. Used to load .h5 files
"""
def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return (1 - SS_res/(SS_tot + K.epsilon()))


"""
Function: smooth_signal        
----------
Use: Apply Savitzy Golay filter 
"""
def smooth_signal(nn):
    nn=nn.astype(float)
    # threshold is hardcoded 11 day auto correlation of MoLS
    threshold=0.925
    nn=savgol_filter(nn,11,3,mode='nearest')
    it=0
    while (pd.Series(nn).autocorr(11) < threshold-0.01) and (it<50):
        nn=savgol_filter(nn,11,3,mode='nearest')
        it+=1
    nn[nn<0]=0
    return nn


"""
Function: format_data        
----------
Use: Function scales input data between [0, 1] and formats into 90-day samples.

Parameters:
  data: a dataframe of daily weather data with columns:
          Counties, Year, Month, Day, Max Temp, Min Temp, Precipitation,
          Humidity, MoLS (replace with 0 if no MoLS data available)
          Data is a .pd file.
  data_shape: shape of one training sample for the model (90x4)

  scaler: MinMaxScaler to scale data to [0,1] and based off of training data.
             Pass previously calculated scaler. Default: None.

  fit_scaler: Boolean argument. If True, function will calculate MinMaxScaler
                 for given data data. Default: False

Returns:
  If fit_scaler = False,
      returns (scaled input data, spatiotemporal information)
  If fit_scaler = True,
      returns (scaled input data, spatiotemporal information, scaler fit to data) 
"""
def format_data(data, data_shape, scaler = None, fit_scaler = False):
    if 'MoLS' not in data.columns:
        data['MoLS']=np.zeros(len(data))
        mols=False
    else:
        mols=True
    data.columns = range(0, len(data.columns))
    if fit_scaler:
        scaler=MinMaxScaler()
        scaler.fit(data.iloc[:, -(data_shape[1] + 1):])
    groups = data.groupby(by = 0)
    X, counties = list(), list()
    for _, subset in groups:
        for i in range(len(subset)):
            if (i+data_shape[0])<len(subset):
                X.append(scaler.transform(subset.iloc[i: i + data_shape[0], -(data_shape[1] + 1):].values))
                counties.append(subset.iloc[i+data_shape[0],0:4])
    return (np.array(X),np.array(counties),mols) if not fit_scaler else (np.array(X),np.array(counties), scaler, mols)

"""
Function: generate_predictions        
----------
Use: Function uses parameter scaler (MinMaxScaler) to transform input data
      in samples of size data_shape, or calculates scaler off of given data.
      Predictions for gravid female mosquitoes are generated using the neural network in
      parameter model.

Input Parameters:
  model: a saved neural network model used for the prediction generation.
          Model is a .h5 file.

  data: a dataframe of daily weather data with columns:
          Counties, Year, Month, Day, Precipitation, Max Temp, Min Temp,
          Humidity, MoLS (replace with 0 if no MoLS data available)
          Data is a .pd file.
          
  data_shape: shape of one training sample for the model (90x4)

  mols: boolean set to False if predictions should be generated without MoLS data

  scaler: MinMaxScaler to scale data to [0,1] and based off of training data.
             Pass previously calculated scaler, otherwise keep default: None.

  fit_scaler: Boolean argument. If True, function will calculate MinMaxScaler
                 for given data data. Default: False

Returns:
  results: Array with the following columns: County, Year, Month, Day,
            MoLS prediction for spatiotemporal location, Model prediction for spatiotemporal location
           (Opt): MinMaxScaler() scaler if fit_scaler=True
"""
def gen_preds(model, data, data_shape, scaler=None, fit_scaler=False, smooth=True):
    # Scale data to [0,1] and reformat to 90-day samples
    if fit_scaler==True:
        X, locs, scaler, mols = format_data(data, data_shape, None, fit_scaler)
    else:
        X, locs, mols = format_data(data, data_shape, scaler, fit_scaler)

    model_preds=np.asarray(model.predict(X[:,:,0:-1]))
    # Scale to original value range and concatenate results
    data_mols=np.zeros((len(model_preds),X.shape[-1]))
    data_mols[:,-1]=X[:,-1,-1]
    data_mols=scaler.inverse_transform(data_mols)

    data_nn=np.zeros((len(model_preds),X.shape[-1]))
    data_nn[:,-1]=model_preds[:,0]
    data_nn=scaler.inverse_transform(data_nn)

    if smooth==True:
        smooth_locs=np.unique(locs[:,0])
        for i in range(0,len(smooth_locs)):
            indices = np.argwhere(locs[:,0]==smooth_locs[i])
            data_nn[indices,-1]=smooth_signal(data_nn[indices,-1][:,0])[:,np.newaxis]
   
    if mols==False:
        results=np.concatenate([locs,np.zeros((len(data_mols),1)),
                       np.reshape(data_nn[:,-1],(len(data_nn),1))], axis=1)
    else:
        results=np.concatenate([locs,np.reshape(data_mols[:,-1],(len(data_mols),1)),
                       np.reshape(data_nn[:,-1],(len(data_nn),1))], axis=1)

    return results if not fit_scaler else (results,scaler)
