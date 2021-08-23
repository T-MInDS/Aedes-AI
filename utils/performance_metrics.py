import os, pdb
import pandas as pd, numpy as np
from sklearn.metrics import r2_score, mean_squared_error, auc
from scipy.stats import pearsonr
from itertools import chain
from glob import glob

#Calculate metrics
def score(mols, nn, return_mse=False):
    r2=r2_score(mols,nn)
    rmse=(mean_squared_error(mols, nn, squared=False))/(max(mols)-min(mols))
    mse=mean_squared_error(mols, nn, squared=True)
    auc_diff=((auc(np.arange(len(mols)),mols)-auc(np.arange(len(nn)),nn))/auc(np.arange(len(mols)),mols))
    pearson=pearsonr(mols,nn)
    return (max(0,r2), rmse, auc_diff, pearson[0], mse) if return_mse==True else (max(0,r2), rmse, auc_diff, pearson[0])

#Calculate mean and standard deviation of metric at location and year level
def gen_perf_metrics(data,seasonal=False,exception=[]):
    groups=data.groupby(['Location','Year'])
    r2s,rmses,auc_diffs,pearsons=list(),list(),list(),list()
    probs=list()
    for group in groups:
        if group[0] not in exception:
            mols=group[1]["MoLS"]
            nn=group[1]["Neural Network"]
            r2,rmse,auc_diff,pearson=score(mols,nn)
            r2s.append(r2)
            rmses.append(rmse)
            auc_diffs.append(auc_diff)
            pearsons.append(pearson)
    r2s=np.asarray(r2s)
    r2s = r2s[np.isfinite(r2s)]
    rmses=np.asarray(rmses)
    rmses = rmses[np.isfinite(rmses)]
    auc_diffs=np.asarray(auc_diffs)
    auc_diffs = auc_diffs[np.isfinite(auc_diffs)]
    pearsons=np.asarray(pearsons)
    pearsons = pearsons[np.isfinite(pearsons)]
    to_return=[np.mean(r2s), np.std(r2s), np.mean(rmses), np.std(rmses),
               np.mean((auc_diffs)), np.std((auc_diffs)), np.mean(pearsons), np.std(pearsons)]
    return to_return

#Calculate metrics at the location and year level
def gen_county_perf_metrics(data,seasonal=False,exception=[]):
    groups=data.groupby(['Location','Year'])
    r2s,rmses,auc_diffs,pearsons=list(),list(),list(),list()
    locs,yrs=list(),list()
    for group in groups:
        if group[0] not in exception:
            mols=group[1]["MoLS"]
            nn=group[1]["Neural Network"]
            r2,rmse,auc_diff,pearson=score(mols,nn)
            #if (np.isfinite(r2) & np.isfinite(rmse) & np.isfinite(auc_diff) & np.isfinite(pearson)):
            if True:
                r2s.append(r2)            
                rmses.append(rmse)
                auc_diffs.append(auc_diff)
                pearsons.append(pearson)
                locs.append(group[0][0])
                yrs.append(group[0][1])
    results=pd.DataFrame()
    results["Location"]=np.asarray(locs)
    results["Year"]=np.asarray(yrs)
    results["R2"]=np.asarray(r2s)
    results["RMSE"]=np.asarray(rmses)
    results["AUC_Diff"]=np.asarray(auc_diffs)
    results["Pearson"]=np.asarray(pearsons)
    return results

# Calculate metrics at the state (location) level
def gen_capitals_perf_metrics(data,seasonal=False,exception=[]):
    groups=data.groupby(['Location'])
    r2s,rmses,auc_diffs,pearsons=list(),list(),list(),list()
    counties,yrs=list(),list()
    for group in groups:
        if group[0] not in exception:
            mols=group[1]["MoLS"]
            nn=group[1]["Neural Network"]
            r2,rmse,auc_diff,pearson=score(mols,nn)
            #if (np.isfinite(r2) & np.isfinite(rmse) & np.isfinite(auc_diff) & np.isfinite(pearson)):
            if True:
                r2s.append(r2)            
                rmses.append(rmse)
                auc_diffs.append(auc_diff)
                pearsons.append(pearson)
                counties.append(group[1].Location.iloc[0])
    results=pd.DataFrame()
    results["Location"]=np.asarray(counties)
    results["R2"]=np.asarray(r2s)
    results["RMSE"]=np.asarray(rmses)
    results["AUC_Diff"]=np.asarray(auc_diffs)
    results["Pearson"]=np.asarray(pearsons)
    return results
