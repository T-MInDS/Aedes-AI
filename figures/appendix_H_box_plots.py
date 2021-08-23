import os, sys
import pandas as pd, numpy as np
from glob import glob
import matplotlib as mpl
import matplotlib.pyplot as plt
import pdb
from sklearn import datasets



if __name__ == '__main__':

    font={'size':16}
    plt.rc('font',**font)

    ddir="./results/"

    global_fil=ddir+"Capitals/capitals_metrics.csv"
    global_data=pd.read_csv(global_fil)
   
    names={'R2':'$R^2_+$', 'RMSE':'$NRMSE$',
           'AUC_Diff':'$Rel.\; AUC\; Diff.$', 'Pearson':'$r$',
           'D_on':'$D_{on}$', 'D_off':'$D_{off}$'}

    cols=['R2', 'RMSE', 'AUC_Diff', 'Pearson']#, 'D_on', 'D_off']
    idx=0
    fig, axs = plt.subplots(2, 2, figsize=(14, 6))
    for col in cols:
        # finding the 1st quartile
        q1 = np.quantile(global_data[col], 0.25)
 
        # finding the 3rd quartile
        q3 = np.quantile(global_data[col], 0.75)
        med = np.median(global_data[col])
 
        # finding the iqr region
        iqr = q3-q1
 
        # finding upper and lower whiskers
        upper_bound = q3+(1.5*iqr)
        lower_bound = q1-(1.5*iqr)
        outliers = global_data[(global_data[col] <= lower_bound) | (global_data[col] >= upper_bound)].Location.values
        print('The following are the outliers in the {} boxplot:{}'.format(col, outliers))
        axs[idx // 2, idx % 2].boxplot(global_data[col], labels=np.asarray(['']), vert=False)#, sym="")
        axs[idx // 2, idx % 2].set_ylabel(names[col])
        idx+=1

    season_fil=ddir+"Threshold_tables/Capitals/"
    
    files=[season_fil+"D_on.csv", season_fil+"D_off.csv"]
    cols=['20%', '40%', '60%', '80%']
    
    fig, axs=plt.subplots(1, 2, figsize=(14,6))
    idx=0
    for fil in files:
        data=pd.read_csv(fil)
        pdb.set_trace()
        data=data.dropna()
        for col in cols:
            # finding the 1st quartile
            q1 = np.quantile(data[col], 0.25)
 
            # finding the 3rd quartile
            q3 = np.quantile(data[col], 0.75)
            med = np.median(data[col])
 
            # finding the iqr region
            iqr = q3-q1
 
            # finding upper and lower whiskers
            upper_bound = q3+(1.5*iqr)
            lower_bound = q1-(1.5*iqr)
            outliers = data[(data[col] <= lower_bound) | (data[col] >= upper_bound)][['Location']].values
            print('The following are the outliers in the {} boxplot:{}'.format(col, np.unique(outliers)))
        dataset=[data['20%'], data['40%'], data['60%'], data['80%']]
        axs[idx].boxplot(dataset, labels=data.columns[-4:], vert=False)#, sym="")
        if 'D_on' in fil:
            axs[idx].set_ylabel(names['D_on'])
        else:
            axs[idx].set_ylabel(names['D_off'])
        idx+=1
    

    plt.show()
        
        
    
