import os
import pandas as pd, numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error, auc
from scipy.stats import pearsonr
from itertools import chain
from glob import glob
import pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils.perf_metrics import *


def autolabel(rects,i,j):
    for rect in rects:
        height=rect.get_height()
        ax[i,j].annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

def gen_map(models, metrics, data):
    heatmap=np.zeros([4, len(models)])
    stds=np.zeros([4,len(models)])
    j=0
    for i in range(0,len(models)):
        subset=data[data.Model==models[i]]
        j=0
        for metric in metrics:
            heatmap[j,i]=np.average(subset[metric])
            stds[j,i]=np.std(subset[metric])
            j+=1
    heatmap=np.around(heatmap,decimals=3)
    stds=np.around(stds,decimals=2)
    return heatmap, stds


if __name__ == '__main__':
    
    fil="./results/Metrics/County_Perf_Metrics.csv"

    data=pd.read_csv(fil)
    metrics=['R2','RMSE','AUC_Diff','Pearson']

    base=np.array(["FF","LSTM","GRU"])

    az_heatmap, az_stds=gen_map(base,metrics,data[data.County.str.contains('Arizona')])
    fl_heatmap, fl_stds=gen_map(base,metrics,data[data.County.str.contains('Florida')])


    font={'size':16}
    plt.rc('font',**font)

    c1=(0.85,0.11,0.38)
    c2=(0.12,0.53,0.90)
    c3=(1,0.76,0.03)
    c4=(0,0.30,0.25)

    x=np.array([0, 0.25, 0.5])
    fig, ax=plt.subplots(2,2)
    rec1=ax[0,0].bar(x,az_heatmap[0,:],yerr=az_stds[0,:],width=0.1,color=c1,label='Arizona')
    rec1=ax[0,0].bar(x+0.1,fl_heatmap[0,:],yerr=fl_stds[0,:],width=0.1,color=c2,label='Florida')
    #autolabel(rec1,0,0)
    ax[0,0].set_ylim([0,1.1])
    ax[0,0].set_xticks(x+0.05)
    ax[0,0].set_xticklabels(["FF","LSTM","GRU"])
    ax[0,0].set_ylabel('$R_+^2$')


    rec1=ax[0,1].bar(x,az_heatmap[-1,:],yerr=az_stds[-1,:],width=0.1,color=c1,label='Arizona')
    rec1=ax[0,1].bar(x+0.1,fl_heatmap[-1,:],yerr=fl_stds[-1,:],width=0.1,color=c2,label='Florida')
    ax[0,1].set_ylim([0,1.1])
    ax[0,1].set_xticks(x+0.05)
    ax[0,1].set_xticklabels(["FF","LSTM","GRU"])
    ax[0,1].set_ylabel('$r$')


    rec1=ax[1,0].bar(x,az_heatmap[1,:],yerr=az_stds[1,:],width=0.1,color=c1,label='Arizona')
    rec1=ax[1,0].bar(x+0.1,fl_heatmap[1,:],yerr=fl_stds[1,:],width=0.1,color=c2,label='Florida')
    #ax[1,0].set_ylim([0,1.1])
    ax[1,0].set_xticks(x+0.05)
    ax[1,0].set_xticklabels(["FF","LSTM","GRU"])
    ax[1,0].set_ylabel('$NRMSE$')


    rec1=ax[1,1].bar(x,az_heatmap[2,:],yerr=az_stds[2,:],width=0.1,color=c1,label='Arizona')
    rec1=ax[1,1].bar(x+0.1,fl_heatmap[2,:],yerr=fl_stds[2,:],width=0.1,color=c2,label='Florida')
    ax[1,1].set_ylim([-0.3,0.7])
    ax[1,1].set_xticks(x+0.05)
    ax[1,1].set_xticklabels(["FF","LSTM","GRU"])
    ax[1,1].set_ylabel('$Rel.\; AUC\; Diff.$')
    ax[1,1].legend(loc='upper right',ncol=2)

    plt.show()
