import os
import pandas as pd, numpy as np
from sklearn.metrics import r2_score, mean_squared_error, auc
from scipy.stats import pearsonr
from itertools import chain
from glob import glob
import pdb
import sys
sys.path.append("./utils")
from perf_metrics import *

def scale_loc_yr():
    files=glob("./results/Test/*")
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
        results=gen_county_perf_metrics(data)
        results["Model"]=""
        results["Subset"]=""
        for i in range(len(results)):
            name=data.iloc[1]["Model"].upper()
            name=name.replace('_MODEL','')
            results.Model.iloc[i]=name
            results.Subset.iloc[i]=subset
        to_save.append(np.asarray(results))
    columns=['County','Year','R2','RMSE','AUC_Diff','Pearson','Model','Subset']
    to_save=list(chain(*to_save))
    to_save=pd.DataFrame(np.asarray(to_save), columns=columns)
    pdb.set_trace()
    to_save.to_csv("./results/Metrics/County_Perf_Metrics.csv",index=False)


def scale_global():
    files=glob("./results/Test/*")
    to_save=list()
    for file in files:
        data=pd.read_csv(file)
        data=data[data.County.str.contains('Arizona')]
        if "Train" in file:
            subset="Train"
        elif "Test" in file:
            subset="Test"
        else:
            subset="Val"
        results=gen_perf_metrics(data)
        results.append(subset)
        results.append(data.iloc[1]["Model"])
        to_save.append(np.asarray(results))
    columns=['Mean_R2','Std_R2','Mean_RMSE','Std_RMSE','Mean_AUC_Diff',
             'Std_AUC_Diff','Mean_Pearson','Std_Pearson','Subset','Model']
    pd.DataFrame(np.asarray(to_save),columns=columns).to_csv(ddir+"Perf_Metrics.csv",index=False)


if __name__ == '__main__':
    scale_global()
    scale_loc_yr()
