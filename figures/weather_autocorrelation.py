import os, json, pdb
import pandas as pd, numpy as np
from scipy.signal import savgol_filter
import sys
sys.path.append("./utils")
sys.path.append("./models")
from generate_predictions import *
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

def main():
    config_file='./models/configs/lstm_config.json'
    scaler_file='./data/data_scaler.gz'
    
    with open(config_file) as fp:
        config = json.load(fp)
   
    data_shape=config['data']['data_shape']
    data_file=config['files']['testing']
    model_file=config['files']['model']

    #load data scaler
    scaler=joblib.load(scaler_file)

    #load data
    data=pd.read_pickle(data_file)
    
    if os.path.exists(model_file):
        model = tf.keras.models.load_model(model_file, custom_objects={'r2_keras':r2_keras})

    X, locs, mols = format_data(data, data_shape, scaler, fit_scaler=False)

    model_preds=np.asarray(model.predict(X[:,:,0:-1]))
    # Scale to original value range and concatenate results
    data_mols=np.zeros((len(model_preds),X.shape[-1]))
    data_mols[:,-1]=X[:,-1,-1]
    data_mols=scaler.inverse_transform(data_mols)

    data_nn=np.zeros((len(model_preds),X.shape[-1]))
    data_nn[:,-1]=model_preds[:,0]
    data_nn=scaler.inverse_transform(data_nn) 
    results=np.concatenate([locs,np.reshape(data_mols[:,-1],(len(data_mols),1)),
                            np.reshape(data_nn[:,-1],(len(data_nn),1))], axis=1)


    results=pd.DataFrame(results, columns=['Location','Year','Month','Day','MoLS','NN'])

    max_lag=21
    lags=np.arange((max_lag))
    grouped=results.groupby('Location')
    nn_corr=np.zeros((max_lag))
    for group in grouped:
        if not group[0]=='Mono,California':
            nn=(group[1].NN).astype(float)
            for lag in lags:
                nn_corr[lag]+=nn.autocorr(lag)
    nn_corr/=(len(grouped)-1)

    lags=np.arange((max_lag))
    data=pd.read_pickle(data_file)
    grouped=data.groupby('Location')
    humidity_corr=np.zeros((max_lag))
    precip_corr=np.zeros((max_lag))
    min_temp_corr=np.zeros((max_lag))
    max_temp_corr=np.zeros((max_lag))
    mols_corr=np.zeros((max_lag))
    for group in grouped:
        humidity=(group[1].Humidity).astype(float)
        precip=(group[1].Precip).astype(float)
        min_temp=(group[1].Min_Temp).astype(float)
        max_temp=(group[1].Max_Temp).astype(float)
        mols=(group[1].MoLS).astype(float)
        for lag in lags:
            humidity_corr[lag]+=humidity.autocorr(lag)
            precip_corr[lag]+=precip.autocorr(lag)
            min_temp_corr[lag]+=min_temp.autocorr(lag)
            max_temp_corr[lag]+=max_temp.autocorr(lag)
            mols_corr[lag]+=mols.autocorr(lag)
    humidity_corr/=len(grouped)
    precip_corr/=len(grouped)
    min_temp_corr/=len(grouped)
    max_temp_corr/=len(grouped)
    mols_corr/=len(grouped)

    font={'size':16}
    plt.rc('font',**font)
    
    plt.figure(0)
    plt.plot(humidity_corr,label='Humidity')
    plt.plot(precip_corr,label='Precip')
    plt.plot(min_temp_corr,label='Min. Temp.')
    plt.plot(max_temp_corr,label='Max. Temp.')
    plt.plot(nn_corr,label='NN')
    plt.vlines(11,-0.1,1.1)
    plt.ylim([-0.05, 1.05])
    plt.xticks(np.arange((max_lag), step=2))
    plt.xlabel('Lag (days)')
    plt.ylabel('Correlation')
    plt.legend()
    plt.show()

 
if __name__ == '__main__':
    main()
