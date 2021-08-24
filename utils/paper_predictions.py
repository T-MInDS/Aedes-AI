import tensorflow as tf
import tensorflow.keras.backend as K
import os, json
import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from itertools import chain
from glob import glob
from scipy.signal import savgol_filter
import sys
sys.path.append("./utils")
sys.path.append("./models")
from predictions import *
import models

def load(config_file, scaler_file, model_files):
    ## Load config, model, data, scaling factors

    #open config file
    with open(config_file) as fp:
        config=json.load(fp)
    data_shape=config["data"]["data_shape"]
    data_files=[config["files"]["training"], config["files"]["validation"], config["files"]["testing"]]

    #load data scaler
    scaler=joblib.load(scaler_file)
    model_files=glob(model_files)
    return data_shape, data_files, scaler, model_files

def predict(model, data, data_shape, scaler, fit_scaler, smooth=True):
    ## Generate abundance forecast
    results=gen_preds(model, data, data_shape, scaler, fit_scaler=False, smooth=smooth)
    return results

def results(data_shape, data_files, scaler, model_files, odir):
    ## Generate raw results
    for model_file in model_files:
        model_name=model_file.split('\\')[-1].split('.')[0]
        model_name=model_name.replace('_model','')
        if "120" in model_name:
            data_shape=[120, 4]
        else:
            data_shape=[90,4]
        #load model
        if os.path.exists(model_file):
            model = tf.keras.models.load_model(model_file,custom_objects={"r2_keras":r2_keras})
            print(model_name+': LOADED')
            #print(model.summary())
            for data_file in data_files:
                subset_name=data_file.split('data/')[-1].split('_data')[0].capitalize()
                #load data
                data=pd.read_pickle(data_file)
                results=predict(model, data, data_shape, scaler, fit_scaler=False)
                print(model_name+': '+subset_name+': '+str(r2_score(results[:,-2],results[:,-1])))
                results=pd.DataFrame(results, columns=["Location","Year","Month","Day","MoLS","Neural Network"])
                results.to_csv(odir+subset_name+'/'+subset_name+"_"+model_name+"_predictions.csv",index=False)   

def raw(data_shape, data_file, scaler, model_file, ofil):
    #load model
    if os.path.exists(model_file):
        model = tf.keras.models.load_model(model_file,custom_objects={"r2_keras":r2_keras})
        print('LOADED')
        #print(model.summary())
        subset_name=data_file.split('data/')[-1].split('_data')[0].capitalize()
        #load data
        data=pd.read_pickle(data_file)
        data=data[(data.Location=='Avondale,Arizona') & (data.Year>2018)]
        results=predict(model, data, data_shape, scaler, fit_scaler=False, smooth=False)
        print(subset_name+': '+str(r2_score(results[:,-2],results[:,-1])))
        results=pd.DataFrame(results, columns=["Location","Year","Month","Day","MoLS","Neural Network"])
        results.to_csv(ofil,index=False)   
     

def main():
    config_file='./models/configs/lstm_config.json'
    scaler_file='./data/data_scaler.gz'
    model_files='./models/saved_models/lstm_model_lo*'
    odir='./results/'

    data_shape, data_files, scaler, model_files = load(config_file, scaler_file, model_files)
    results(data_shape, data_files, scaler, model_files, odir)
    models=['FF','LSTM','GRU']
    for model in models:
        raw(data_shape=data_shape, data_file='./data/test_data.pd', scaler=scaler,
                model_file='./models/saved_models\\{}_model.h5'.format(model.lower()),
                ofil=odir+'Test/Test_{}_raw_predictions.csv'.format(model.lower()))

if __name__ == '__main__':
    main()
