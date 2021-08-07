import tensorflow as tf
import tensorflow.keras.backend as K
import os, json, sys, math
import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from itertools import chain
from glob import glob
from scipy.signal import savgol_filter
sys.path.append("./utils")
sys.path.append("./models")
from predictions import *
from performance_metrics import *

def load(config_file, scaler_file):
    ## Load config, model, data, scaling factors

    #open config file
    with open(config_file) as fp:
        config=json.load(fp)
    data_shape=config["data"]["data_shape"]

    #load data scaler
    scaler=joblib.load(scaler_file)
    model_file=config['files']['model']
    
    return data_shape, scaler, model_file

def predict(model, data, data_shape, scaler, model_name, fit_scaler):
    ## Generate abundance forecast
    results=gen_preds(model, data, data_shape, scaler, fit_scaler=False)
    return results

def results(data_shape, data_file, scaler, model_file, odir):
    ## Generate raw results
    model_name=model_file.split('/')[-1].split('.')[0]
    model_name=model_name.replace('_model','')
    #load model
    if os.path.exists(model_file):
        model = tf.keras.models.load_model(model_file,custom_objects={"r2_keras":r2_keras})
        print(model_name+': LOADED')
        #print(model.summary())
        #load data
        data=pd.read_pickle(data_file)
        results=predict(model, data, data_shape, scaler, model_name, fit_scaler=False)
        print(model_name+': '+str(r2_score(results[:,-2],results[:,-1])))
        results=pd.DataFrame(results, columns=["Location","Year","Month","Day","MoLS","Neural Network"])
        results.to_csv(odir+"Capitals_gru_hi_predictions.csv",index=False)

def scale_loc_yr(file, fname):
    data=pd.read_csv(file)
    data=data[data.Year>2011]
    score_results=gen_county_perf_metrics(data)
    columns=['Location','Year','R2','RMSE','AUC_Diff','Pearson']
    to_save=pd.DataFrame(score_results, columns=columns)
    to_save.to_csv(fname,index=False)

def peaks(file):
    os.system("python utils/match_peaks.py -r "+file)

def combined_score():
    ddir='./results/'
    global_fit=pd.read_csv(ddir+"Capitals/capitals_scores.csv")
    locs=np.unique(global_fit.Location)
    thres_mo_off=pd.read_csv(ddir+"Threshold_tables/Capitals/gru_hi_D_off_table.csv")
    thres_mo_on=pd.read_csv(ddir+"Threshold_tables/Capitals/gru_hi_D_on_table.csv")

    ds=np.zeros((len(locs)))
    hs=np.zeros((len(locs)))
    for i in range(0, len(locs)):
        subset=global_fit[global_fit.Location==locs[i]]
        m1=(1-np.sqrt(subset.R2**2*subset.Pearson**2))
        m2=np.sqrt(subset.AUC_Diff**2+subset.RMSE**2)
        d=np.average(np.sqrt(m1**2+m2**2))
        ds[i]=d

        thres_off=thres_mo_off[(thres_mo_off.City.astype('str')==locs[i].split(',')[0]) & (thres_mo_off.State==locs[i].split(',')[-1])]
        thres_on=thres_mo_on[(thres_mo_on.City.astype('str')==locs[i].split(',')[0]) & (thres_mo_on.State==locs[i].split(',')[-1])]
        Z=(1+(thres_on[["20%","40%","60%","80%"]].isna()*1).sum(axis=0)/len(thres_on))
        for col in ["20%","40%","60%","80%"]:
            if thres_on[col].isna().any():
                thres_on[col].fillna(np.mean(np.absolute(thres_mo_on[col])), inplace=True)
            if thres_off[col].isna().any():
                thres_off[col].fillna(np.mean(np.absolute(thres_mo_off[col])), inplace=True)
            ht=1/sum(Z)*sum(Z*np.sqrt(np.mean(np.absolute(thres_on[["20%","40%","60%","80%"]]))*
                            np.std(np.absolute(thres_on[["20%","40%","60%","80%"]]))+
                            np.mean(np.absolute(thres_off[["20%","40%","60%","80%"]]))*
                            np.std(np.absolute(thres_off[["20%","40%","60%","80%"]]))))
            if math.isnan(ht):
                print(locs[i])
            hs[i]=ht
    ds=ds/np.nanmax(ds)
    hs=hs/np.nanmax(hs)
    S=np.sqrt(ds**2+hs**2)
    S=pd.DataFrame(S,index=locs)
    pdb.set_trace()
    S.to_csv(ddir+"Capitals/combined_scores.csv")
    return 

def main():
    config_file='./models/configs/gru_config_hi.json'
    scaler_file='./data/data_scaler.gz'
    data_file='./data/capitals.pd'
    odir='./results/Capitals/'

    data_shape, scaler, model_file = load(config_file, scaler_file)
    #results(data_shape, data_file, scaler, model_file, odir)

    #scale_loc_yr(odir+'Capitals_gru_hi_predictions.csv', odir+'capitals_scores.csv')

    #peaks(odir+'Capitals_gru_hi_predictions.csv')

    combined_score()

if __name__ == '__main__':
    main()
