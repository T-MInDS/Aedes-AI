import os
import pandas as pd, numpy as np
from sklearn.metrics import r2_score, mean_squared_error, auc
from scipy.stats import pearsonr
from itertools import chain
from glob import glob
import pdb
import sys
sys.path.append("./utils")
from performance_metrics import *

def scale_loc_yr(files, fname):
    to_save=list()
    for file in files:
        if "Train" in file:
            subset="Train"
        elif "Test" in file:
            subset="Test"
        else:
            subset="Val"
        data=pd.read_csv(file)
        data=data[data.Year>2011]
        score_results=gen_county_perf_metrics(data)
        score_results["Model"]=""
        score_results["Subset"]=""
        for i in range(len(score_results)):
            name=file.split('Test_')[-1].split('_predictions')[0].upper()
            score_results.Model.iloc[i]=name
            score_results.Subset.iloc[i]=subset
        to_save.append(np.asarray(score_results))
    columns=['Location','Year','R2','RMSE','AUC_Diff','Pearson','Model','Subset']
    to_save=list(chain(*to_save))
    to_save=pd.DataFrame(np.asarray(to_save), columns=columns)
    to_save.to_csv(fname,index=False)


def scale_global_means(files, fname):    
    to_save=list()
    for file in files:
        data=pd.read_csv(file)
        #data=data[data.County.str.contains('Arizona')]
        if "Train" in file:
            subset="Train"
        elif "Test" in file:
            subset="Test"
        else:
            subset="Val"
        model=file.split(subset+'_')[-1].split('_predictions')[0]
        results=gen_perf_metrics(data)
        results.append(subset)
        results.append(model)
        to_save.append(np.asarray(results))
    columns=['Mean_R2','Std_R2','Mean_RMSE','Std_RMSE','Mean_AUC_Diff',
             'Std_AUC_Diff','Mean_Pearson','Std_Pearson','Subset','Model']
    pd.DataFrame(np.asarray(to_save),columns=columns).to_csv(fname,index=False)

def scale_global(files, fname):
    to_save=list()
    for file in files:
        data=pd.read_csv(file)
        if "Train" in file:
            subset="Train"
        elif "Test" in file:
            subset="Test"
        else:
            subset="Val"
        model=file.split(subset+'_')[-1].split('_predictions')[0]
        r2, rmse, auc_diff, pearson = score(data["MoLS"], data["Neural Network"])
        results=[r2, rmse, auc_diff, pearson, subset, model]
        to_save.append(np.asarray(results))
    columns=["R2","RMSE","AUC_Diff","Pearson","Subset","Model"]
    pd.DataFrame(np.asarray(to_save), columns=columns).to_csv(fname, index=False)

if __name__ == '__main__':
    test_files=glob("./results/Test/*")
    val_files=glob("./results/Val/*")
    train_files=glob("./results/Train/*")
    files=[*test_files, *val_files, *train_files]

    ddir='./results/Metrics/'
    scale_global(files, ddir+'Global_scores.csv')
    scale_global_means(files, ddir+'Global_mean_scores.csv')
    scale_loc_yr(files, ddir+'Location_scores.csv')
    
