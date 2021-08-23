import os, sys, joblib
import pandas as pd, numpy as np
from itertools import chain
from glob import glob
import math, pdb
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
sys.path.append('./models')
from training import format_data, split_and_shuffle



def smoothing_corr():
    data = pd.read_pickle('./data/train_data.pd')
    grouped = data.groupby(['Location'])

    lags = np.arange(0,20)

    plt_data = np.zeros([len(lags), 4])

    for lag in lags:
        for group in grouped:
            max_temp = group[1].Max_Temp.astype('float')
            min_temp = group[1].Min_Temp.astype('float')
            precip = group[1].Precip.astype('float')
            humidity = group[1].Humidity.astype('float')
            #mols = group[1].MoLS.astype('float')
            plt_data[lag,0]+=max_temp.autocorr(lag=lag)
            plt_data[lag,1]+=min_temp.autocorr(lag=lag)
            plt_data[lag,2]+=precip.autocorr(lag=lag)
            plt_data[lag,3]+=humidity.autocorr(lag=lag)
            #plt_data[lag,4]+=mols.autocorr(lag=lag)


    plt_data/=len(grouped)

    plt.figure()
    plt.plot(plt_data)
    plt.xticks(np.arange(0,len(lags),2))
    plt.xlabel('Lag (Days)')
    plt.ylabel('Average Sample Autocorrelation')
    plt.legend(['Max. Temp.', 'Min. Temp.', 'Precip.', 'RH'])#, 'MoLS'])
    plt.show()
    return


def shuffling_corr():
    data = pd.read_pickle('./data/train_data.pd')
    scaler = joblib.load('./data/data_scaler.gz')

    data = format_data(data, data_shape=[90,4], samples_per_city=2200, scaler=scaler)
    X, y = split_and_shuffle(data)
    X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))

    max_temp = pd.Series(X[:,0])
    min_temp = pd.Series(X[:,1])
    precip = pd.Series(X[:,2])
    humidity = pd.Series(X[:,3])

    max_lag = 360

    plt_data = np.zeros([max_lag, 4])

    for lag in tqdm(range(0, max_lag), desc='Progress'):
        #mols = group[1].MoLS.astype('float')
        plt_data[lag,0]=max_temp.autocorr(lag=lag)
        plt_data[lag,1]=min_temp.autocorr(lag=lag)
        plt_data[lag,2]=precip.autocorr(lag=lag)
        plt_data[lag,3]=humidity.autocorr(lag=lag)
        #plt_data[lag,4]+=mols.autocorr(lag=lag)

    plt.figure()
    plt.plot(plt_data)
    plt.xticks(np.arange(0,max_lag,50))
    plt.xlabel('Lag (Days)')
    plt.ylabel('Autocorrelation')
    plt.legend(['Max. Temp.', 'Min. Temp.', 'Precip.', 'RH'])
    plt.show()
    
    return

if __name__ == '__main__':
    font={'size':16}
    plt.rc('font',**font)
    smoothing_corr()

    

