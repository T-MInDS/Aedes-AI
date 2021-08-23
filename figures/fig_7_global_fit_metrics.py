import os, sys
import pandas as pd, numpy as np
from glob import glob
import pdb
import matplotlib as mpl
import matplotlib.pyplot as plt

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
    stds=np.around(stds,decimals=3)
    return heatmap, stds


if __name__ == '__main__':
    
    fil="./results/Metrics/Location_scores.csv"
    
    data=pd.read_csv(fil)
    metrics=['R2','RMSE','AUC_Diff','Pearson']

    base=np.array(["FF","LSTM","GRU"])
    hi=np.array(["FF_HI","LSTM_HI","GRU_HI"])
    lo=np.array(["FF_LO","LSTM_LO","GRU_LO"])
    both=np.array(["FF_HI_LO","LSTM_HI_LO","GRU_HI_LO"])
                 
    base_mean, base_stds=gen_map(base,metrics,data[data.Subset=='Test'])
    hi_mean, hi_stds=gen_map(hi,metrics,data[data.Subset=='Test'])
    lo_mean, lo_stds=gen_map(lo,metrics,data[data.Subset=='Test'])
    both_mean, both_stds=gen_map(both,metrics,data[data.Subset=='Test'])
    

    font={'size':16}
    plt.rc('font',**font)

    c1=(0.85,0.11,0.38)
    c2=(0.12,0.53,0.90)
    c3=(1,0.76,0.03)
    c4=(0,0.30,0.25)

    x=np.array([0, 0.45, 0.9])
    fig, ax=plt.subplots(2,2)

    orders=[0,-1,1,2]
    count=0
    for i in range(0,2):
        for j in range(0,2):
            ax[i,j].bar(x,base_mean[orders[count],:],yerr=base_stds[orders[count],:],width=0.1,color=c1,label='Base')
            ax[i,j].bar(x+0.1,hi_mean[orders[count],:],yerr=hi_stds[orders[count],:],width=0.1,color=c2,label='HI')
            ax[i,j].bar(x+0.2,lo_mean[orders[count],:],yerr=lo_stds[orders[count],:],width=0.1,color=c3,label='LO')
            ax[i,j].bar(x+0.3,both_mean[orders[count],:],yerr=both_stds[orders[count],:],width=0.1,color=c4,label='HI LO')
            ax[i,j].set_xticks(x+0.15)
            ax[i,j].set_xticklabels(["FF","LSTM","GRU"])
            count+=1

    ax[0,0].set_ylim([0,1.1])
    ax[0,0].set_ylabel('$R_+^2$')

    ax[0,1].set_ylim([0,1.1])
    ax[0,1].set_ylabel('$r$')

    ax[1,0].set_ylabel('$NRMSE$')

    ax[1,1].set_ylim([-0.4,0.36])
    ax[1,1].set_ylabel('$Rel.\; AUC\; Diff.$')
    ax[1,1].legend(loc='lower right',ncol=4)

    plt.show()
