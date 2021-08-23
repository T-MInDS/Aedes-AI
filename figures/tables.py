import os, sys
import pandas as pd, numpy as np
from glob import glob
from sklearn.metrics import mean_squared_error as rmse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
import pdb


def tab_2():
    outfil=open('./results/Metrics/tab_2.txt','w+')
    
    models=['LR', 'FF', 'LSTM', 'GRU']
    subsets=['Train', 'Val', 'Test']
    names={'LR':'Baseline','FF':'FF','LSTM':'LSTM','GRU':'GRU'}

    for model in models:
        outfil.write('\t\\hline \n')
        rmses, r2s=list(), list()
        for subset in subsets:
            file='./results/'+subset+'/'+subset+'_'+model.lower()+'_predictions.csv'
            data=pd.read_csv(file)
            data=data[data.Year>2011]
            mols=data['MoLS']
            nn=data['Neural Network']
            r2s.append(r2_score(mols, nn))
            rmses.append(rmse(mols, nn, squared=False))
            print('{} {}: {}'.format(model, subset, mae(mols, nn)))
        formatter=lambda d: str(round(d,3))
        rstring='\t\\multirow{2}{*}{'+names[model]+'} & $RMSE$ & '
        rstring+=' & '.join(map(formatter,rmses))
        outfil.write(rstring+'\\\ \n')
        rstring='\t& $R^2$ & '
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
        outfil.write('\t\\hline \n')
        row=data.iloc[i,:]
        formatter=lambda d: str(round(d[0], 3)) + ' $\\pm$ ' + str(round(d[1],3))
        means=row.filter(like='Mean').squeeze()
        stds=row.filter(like='Std').squeeze()
        rstring='\t' + names[row.Model.upper()] + ' & '
        rstring+=' & '.join(map(formatter, zip(means, stds)))
        outfil.write(rstring+'\\\ \n\n')
    outfil.close()



def tab_4():
    table_bases=['ff','lstm','gru']
    means=pd.read_csv('./results/Threshold_tables/Test/fl_mean.csv')
    means=means[means.Model.isin(table_bases)].reset_index(drop=True)
    
    stds=pd.read_csv('./results/Threshold_tables/Test/fl_std.csv')
    stds=stds[stds.Model.isin(table_bases)].reset_index(drop=True)

    outfil=open('./results/Metrics/tab_4_fl.txt','w+')
    cols=['20%','40%','60%','80%']

    order=means.copy()
    order[cols]=order[cols].abs().multiply(stds[cols])
    print('D_on order')
    print(order.Model.iloc[order[order.Metric=='D_on'][cols].idxmin()])
    print('D_off order')
    print(order.Model.iloc[order[order.Metric=='D_off'][cols].idxmin()])        
    
    for base in table_bases:
        outfil.write('\t\\hline \n')
        formatter=lambda d: str(round(d[0], 3)) + ' $\\pm$ ' + str(round(d[1],3))
        mean=means[means.Model==base]
        std=stds[stds.Model==base]
        rstring='\t\\multirow{2}{*}{' + base.upper() + '} & $D_{on}$ & '
        rstring+=' & '.join(map(formatter, zip(mean[mean.Metric=='D_on'][cols].iloc[0], std[std.Metric=='D_on'][cols].iloc[0])))
        outfil.write(rstring+'\\\ \n')

        rstring='\t& $D_{off}$ & '
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
    means=pd.read_csv('./results/Threshold_tables/Test/collier_mean.csv')
    means=means[means.Model.isin(table_bases)].reset_index(drop=True)
    
    outfil=open('./results/Metrics/tab_12_collier.txt','w+')
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

def tab_13():
    infil='./results/Metrics/Global_mean_scores.csv'

    data=pd.read_csv(infil)
    data=data[data.Subset=='Test']

    models=['FF', 'LSTM', 'GRU']
    for model in models:
        table_bases=[model, model+'\'', model+'_2000', model+'_120']
        names={model:model, model+'\'':model+'$\'$',
               model+'_2000':model+'$_{2000}$', model+'_120':model+'$_{120}$'}

        outfil=open('./results/Metrics/tab_13_{}.txt'.format(model.lower()),'w+')

        for base in table_bases:
            outfil.write('\t \\hline \n')
            subset=data[data.Model==base.lower()]
            formatter=lambda d: str(round(d[0], 3)) + ' $\\pm$ ' + str(round(d[1],3))
            means=subset.filter(like='Mean').squeeze()
            stds=subset.filter(like='Std').squeeze()
            rstring='\t ' + names[base] + ' & '
            rstring+=' & '.join(map(formatter, zip(means, stds)))
            outfil.write(rstring+'\\\ \n\n')
        outfil.close()


def tab_14():
    models=['FF', 'LSTM', 'GRU']
    cols=['20%','40%','60%','80%']
    
    means=pd.read_csv('./results/Threshold_tables/Test/all_mean.csv')        
    stds=pd.read_csv('./results/Threshold_tables/Test/all_std.csv')

    for model in models:
        table_bases=[model, model+'\'', model+'_2000', model+'_120']
        names={model:model, model+'\'':model+'$\'$',
               model+'_2000':model+'$_{2000}$', model+'_120':model+'$_{120}$'}

        outfil=open('./results/Metrics/tab_14_{}.txt'.format(model.lower()),'w+')
        
        for base in table_bases:
            outfil.write('\t \\hline \n')
            formatter=lambda d: str(round(d[0], 3)) + ' $\\pm$ ' + str(round(d[1],3))
            mean=means[means.Model==base.lower()]
            std=stds[stds.Model==base.lower()]
            rstring='\t \\multirow{2}{*}{' + names[base] + '} & $D_{on}$ & '
            rstring+=' & '.join(map(formatter, zip(mean[mean.Metric=='D_on'][cols].iloc[0], std[std.Metric=='D_on'][cols].iloc[0])))
            outfil.write(rstring+'\\\ \n')

            rstring='\t & $D_{off}$ & '
            rstring+=' & '.join(map(formatter, zip(mean[mean.Metric=='D_off'][cols].iloc[0], std[std.Metric=='D_off'][cols].iloc[0])))
            outfil.write(rstring+'\\\ \n\n')
        outfil.close()
    


if __name__ == '__main__':
    tab_14()
