import os, sys
import pandas as pd, numpy as np
from glob import glob
from sklearn.metrics import mean_squared_error as rmse
from sklearn.metrics import r2_score
import pdb


def tab_2():
    outfil=open('./results/Metrics/tab_2.txt','w+')
    
    models=['LR', 'FF', 'LSTM', 'GRU']
    subsets=['Train', 'Val', 'Test']
    names={'LR':'Baseline','FF':'FF','LSTM':'LSTM','GRU':'GRU'}

    for model in models:
        outfil.write('\\hline \n')
        rmses, r2s=list(), list()
        for subset in subsets:
            file='./results/'+subset+'/'+subset+'_'+model.lower()+'_predictions.csv'
            data=pd.read_csv(file)
            mols=data['MoLS']
            nn=data['Neural Network']
            r2s.append(r2_score(mols, nn))
            rmses.append(rmse(mols, nn, squared=False))
        formatter=lambda d: str(round(d,3))
        rstring='\\multirow{2}{*}{'+names[model]+'} & $RMSE$ & '
        rstring+=' & '.join(map(formatter,rmses))
        outfil.write(rstring+'\\\ \n')
        rstring='& $R^2$ & '
        rstring+=' & '.join(map(formatter,r2s))
        outfil.write(rstring+'\\\ \n\n')
    outfil.close()

def tab_3():
    outfil=open('./results/Metrics/tab_3.txt','w+')
    
    models=['LR', 'FF', 'LSTM', 'GRU']
    subsets=['Train', 'Val', 'Test']
    names={'LR':'Baseline','FF':'FF','LSTM':'LSTM','GRU':'GRU'}

    infil='./results/Metrics/Global_mean_scores.csv'
    
    data=pd.read_csv(infil)
    data=data[data.Subset=='Test']
    data=data[data.Model.isin([model.lower() for model in models])].reset_index(drop=True)
    for i in range(0,len(data)):
        outfil.write('\\hline \n')
        row=data.iloc[i,:]
        formatter=lambda d: str(round(d[0], 3)) + ' $\\pm$ ' + str(round(d[1],3))
        means=row.iloc[[0,2,4,6]]
        stds=row.iloc[[1,3,5,7]]
        rstring=names[row.Model.upper()] + ' & '
        rstring+=' & '.join(map(formatter, zip(means, stds)))
        outfil.write(rstring+'\\\ \n\n')
    outfil.close()


def tab_4():
    table_bases=['ff','lstm','gru']
    means=pd.read_csv('./results/Threshold_tables/Test/az_mean.csv')
    means=means[means.Model.isin(table_bases)].reset_index(drop=True)
    
    stds=pd.read_csv('./results/Threshold_tables/Test/az_std.csv')
    stds=stds[stds.Model.isin(table_bases)].reset_index(drop=True)

    #outfil=open('./results/Metrics/tab_4_fl.txt','w+')
    cols=['20%','40%','60%','80%']

    order=means.copy()
    order[cols]=order[cols].abs().multiply(stds[cols])
    print('D_on order')
    print(order.Model.iloc[order[order.Metric=='D_on'][cols].idxmin()])
    print('D_off order')
    print(order.Model.iloc[order[order.Metric=='D_off'][cols].idxmin()])        
    
    for base in table_bases:
        outfil.write('\\hline \n')
        formatter=lambda d: str(round(d[0], 3)) + ' $\\pm$ ' + str(round(d[1],3))
        mean=means[means.Model==base]
        std=stds[stds.Model==base]
        rstring='\\multirow{2}{*}{' + base.upper() + '} & $D_{on}$ & '
        rstring+=' & '.join(map(formatter, zip(mean[mean.Metric=='D_on'][cols].iloc[0], std[std.Metric=='D_on'][cols].iloc[0])))
        outfil.write(rstring+'\\\ \n')

        rstring='& $D_{off}$ & '
        rstring+=' & '.join(map(formatter, zip(mean[mean.Metric=='D_off'][cols].iloc[0], std[std.Metric=='D_off'][cols].iloc[0])))
        outfil.write(rstring+'\\\ \n\n')
    outfil.close()

def tab_11():
    table_bases=['ff','lstm','gru',
                 'ff_hi','ff_lo','ff_hi_lo',
                 'lstm_hi','lstm_lo','lstm_hi_lo',
                 'gru_hi','gru_lo','gru_hi_lo']
    means=pd.read_csv('./results/Threshold_tables/Test/all_mean.csv')
    means=means[means.Model.isin(table_bases)].reset_index(drop=True)
    
    stds=pd.read_csv('./results/Threshold_tables/Test/all_std.csv')
    stds=stds[stds.Model.isin(table_bases)].reset_index(drop=True)

    outfil=open('./results/Metrics/tab_11.txt','w+')
    cols=['20%','40%','60%','80%']

    order=means.copy()
    order[cols]=order[cols].abs().multiply(stds[cols])
    print('D_on order')
    print(order.Model.iloc[order[order.Metric=='D_on'][cols].idxmin()])
    print('D_off order')
    print(order.Model.iloc[order[order.Metric=='D_off'][cols].idxmin()])        
    
    for base in table_bases:
        outfil.write('\t \\hline \n')
        formatter=lambda d: str(round(d[0], 3)) + ' $\\pm$ ' + str(round(d[1],3))
        mean=means[means.Model==base]
        std=stds[stds.Model==base]
        rstring='\t \\multirow{2}{*}{' + base.upper() + '} & $D_{on}$ & '
        rstring+=' & '.join(map(formatter, zip(mean[mean.Metric=='D_on'][cols].iloc[0], std[std.Metric=='D_on'][cols].iloc[0])))
        outfil.write(rstring+'\\\ \n')

        rstring='\t & $D_{off}$ & '
        rstring+=' & '.join(map(formatter, zip(mean[mean.Metric=='D_off'][cols].iloc[0], std[std.Metric=='D_off'][cols].iloc[0])))
        outfil.write(rstring+'\\\ \n\n')
    outfil.close()
    
def tab_12():
    table_bases=['gru','gru_hi','gru_lo','gru_hi_lo']
    means=pd.read_csv('./results/Threshold_tables/Test/avondale_mean.csv')
    means=means[means.Model.isin(table_bases)].reset_index(drop=True)
    
    outfil=open('./results/Metrics/tab_12_avondale.txt','w+')
    cols=['20%','40%','60%','80%']

    names={'gru':'Base','gru_hi':'HI','gru_lo':'LO','gru_hi_lo':'HI LO'}

    for base in table_bases:
        outfil.write('\t \\hline \n')
        formatter=lambda d: str(round(d, 3))
        mean=means[means.Model==base]
        rstring='\t \\multirow{2}{*}{' + names[base] + '} & $D_{on}$ & '
        rstring+=' & '.join(map(formatter, mean[mean.Metric=='D_on'][cols].iloc[0]))
        outfil.write(rstring+'\\\ \n')

        rstring='\t & $D_{off}$ & '
        rstring+=' & '.join(map(formatter, mean[mean.Metric=='D_off'][cols].iloc[0]))
        outfil.write(rstring+'\\\ \n\n')
    outfil.close()    

if __name__ == '__main__':
    tab_3()
