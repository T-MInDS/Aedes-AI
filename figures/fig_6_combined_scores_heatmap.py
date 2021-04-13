import os
import pandas as pd, numpy as np
from itertools import chain
from glob import glob
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.perf_metrics import *


def sort_state(data,state):
    subset=data.filter(regex=state)
    subset=(subset.mean(axis=0)).sort_values(axis=0,ascending=False)
    return subset.index

def sort_models(data):
    data=(data.mean(axis=1)).sort_values(axis=0,ascending=True)
    return data.index

def create_heatmap(heatmap, states, x_labels, y_labels):
    cmap='viridis'
    font={'size':16}
    mpl.rc('font',**font)

    plt.figure(figsize=(20,10))
    plt.imshow(max(heatmap.max())-heatmap, cmap=cmap)
    plt.clim(min(heatmap.min()),max(heatmap.max()))
    plt.xticks(ticks=np.arange(heatmap.shape[1]), labels=(['']*len(x_labels)))
    plt.yticks(ticks=np.arange(heatmap.shape[0]), labels=np.char.replace(np.asarray(y_labels,dtype='str'),'_',' '))
    cbar=plt.colorbar(ticks=np.arange(min(heatmap.min())+(max(heatmap.max())-1),max(heatmap.max()),0.2),
                 aspect=50, fraction=0.165, pad=0.02, orientation='horizontal')
    cbar.set_ticklabels(['1.0','0.8','0.6','0.4','0.2'])
    plt.tight_layout()
    ln=-0.5
    state_dict={'Arizona':'AZ', 'California':'CA', 'Florida':'FL',
                'Wisconsin':'WI', 'Texas':'TX', 'New York':'NY',
                'Connecticut':'CT', 'North Carolina':'NC', 'New Jersey':'NJ'}
    for state in states:
        nxt=(heatmap.filter(regex=state)).shape[1]
        ln+=nxt
        plt.text(ln-nxt/2-0.2,-0.6,state_dict[state])
        if ln<(heatmap.shape[1]-1):
            plt.axvline(x=ln,linewidth=3,color='w')
    #plt.savefig(ddir+"combined_heatmap.png",bbox_inches='tight', dpi=600)
    plt.show()
 
    
def combined_metric():
    ddir="./results/Metrics/"
    global_fit=pd.read_csv(ddir+"County_Perf_Metrics.csv")
    global_fit=global_fit[global_fit.Subset=="Test"]
    locs=np.unique(global_fit.County)
    models=np.unique(global_fit.Model)
    ds=np.zeros((len(models),len(locs)))
    hs=np.zeros((len(models),len(locs)))
    for i in range(0,len(models)):
        thres_mo_off=pd.read_csv(ddir+"Threshold_tables/"+models[i].lower()+"_D_off_table.csv")
        thres_mo_on=pd.read_csv(ddir+"Threshold_tables/"+models[i].lower()+"_D_on_table.csv")
        for j in range(0,len(locs)):
            subset=global_fit[(global_fit.Model==models[i]) & (global_fit.County==locs[j])]
            m1=(1-np.sqrt(subset.R2**2*subset.Pearson**2))
            m2=np.sqrt(subset.AUC_Diff**2+subset.RMSE**2)
            d=np.average(np.sqrt(m1**2+m2**2))
            ds[i,j]=d
            thres_off=thres_mo_off[(thres_mo_off.City==locs[j].split(',')[0]) & (thres_mo_off.State==locs[j].split(',')[-1])]
            thres_on=thres_mo_on[(thres_mo_on.City==locs[j].split(',')[0]) & (thres_mo_on.State==locs[j].split(',')[-1])]
            Z=(1+(thres_on[["20%","40%","60%","80%"]].isna()*1).sum(axis=0)/len(thres_on))
            for col in ["20%","40%","60%","80%"]:
                if thres_on[col].isna().all():
                    thres_on[col]=thres_on[col].fillna(np.mean(np.absolute(thres_mo_on[col])))
                if thres_off[col].isna().all():
                    thres_off[col]=thres_off[col].fillna(np.mean(np.absolute(thres_mo_off[col])))
            ht=1/sum(Z)*sum(Z*np.sqrt(np.mean(np.absolute(thres_on[["20%","40%","60%","80%"]]))*
                                np.std(np.absolute(thres_on[["20%","40%","60%","80%"]]))+
                                np.mean(np.absolute(thres_off[["20%","40%","60%","80%"]]))*
                                np.std(np.absolute(thres_off[["20%","40%","60%","80%"]]))))
            if math.isnan(ht):
                pdb.set_trace()
            hs[i,j]=ht
    ds=ds/ds.max()
    hs=hs/hs.max()
    S=np.sqrt(ds**2+hs**2)
    S=pd.DataFrame(S,columns=locs,index=models)
    #S.to_csv(ddir+"combined_scores.csv")
    return S

if __name__ == '__main__':

    scores=combined_metric()
    states=["Arizona","Wisconsin","Texas","California","New York","Connecticut",
            "North Carolina","New Jersey","Florida"]
    loc_order=list()
    for state in states:
        order=sort_state(scores,state)
        loc_order.append(order)
    loc_order=list(chain(*loc_order))
    mo_order=sort_models(scores)
    scores=scores[loc_order].loc[mo_order]
    create_heatmap(scores, states, loc_order, mo_order)
