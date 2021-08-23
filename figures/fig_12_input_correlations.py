import os, sys, pdb
import pandas as pd, numpy as np
from matplotlib.colors import Normalize, rgb2hex, LogNorm
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import pearsonr
from sklearn.metrics import r2_score


def score_avg_mos_corr():
    data = pd.read_csv('./results/Capitals/Capitals_gru_hi_predictions.csv')
    data2 = pd.read_csv('./results/Capitals/capitals_combined_scores.csv')
    data2 = data2.rename(columns={'Unnamed: 0':'State'})
    data = data[data.Location.str.contains('45')]

    scores, avgs = list(), list()
    for state in np.unique(data.Location):
        try:
            scores.append(data2[data2.State==state].Score.iloc[0])
            avgs.append(np.mean(data[data.Location==state].MoLS))
        except:
            pass
    avgs = np.asarray(avgs)
    scores = np.asarray(scores)

    avg_norm = LogNorm(vmin=avgs.min(), vmax=avgs.max())
    score_norm = LogNorm(vmin=scores.min(), vmax=scores.max())

    avgs = avg_norm(avgs)
    scores=score_norm(scores)
    
    corr, _ = pearsonr(avgs, scores)
    r2 = r2_score(avgs, scores)
    print('Pearson correlation: {}'.format(corr))
    print('R^2: {}'.format(r2))


def samples_corr():
    capitals = pd.read_pickle('./data/capitals.pd')
    capitals = capitals[~capitals.Location.str.contains('85')]
    capitals = capitals[~capitals.Location.str.contains('California')]
    capitals = capitals[~capitals.Location.str.contains('Arizona')]
    capitals = capitals[~capitals.Location.str.contains('Texas')]
    capitals = capitals[~capitals.Location.str.contains('Wisconsin')]
    capitals = capitals[~capitals.Location.str.contains('Minnesota')]
    capitals = capitals[~capitals.Location.str.contains('North Carolina')]
    capitals = capitals[~capitals.Location.str.contains('Delaware')]
    capitals = capitals[~capitals.Location.str.contains('New Jersey')]
    capitals = capitals[capitals.Year==2016].reset_index(drop=True)
    capitals['Avg_Temp'] = capitals[['Max_Temp','Min_Temp']].mean(axis=1)
    
    train = pd.read_pickle('./data/train_data.pd')
    train = train[train.Year==2016].reset_index(drop=True)
    train['Avg_Temp'] = train[['Max_Temp','Min_Temp']].mean(axis=1)
    
    test = pd.read_pickle('./data/test_data.pd')
    test = test[test.Year==2016].reset_index(drop=True)
    test['Avg_Temp'] = test[['Max_Temp','Min_Temp']].mean(axis=1)
    
    capitals_group = capitals[['Location','Avg_Temp','Precip','MoLS']].groupby(['Location'])
    train_group = train[['Location','Avg_Temp','Precip','MoLS']].groupby(['Location'])
    test_group = test[['Location','Avg_Temp','Precip','MoLS']].groupby(['Location'])

    fig, axs = plt.subplots(1,3)
    names = ['Avg_Temp', 'Precip', 'MoLS']
    name_dic = {'Avg_Temp':'Average Temperature', 'Precip':'Precipitation', 'MoLS':'MoLS'}
    for i in range(0,3):
        corr_data = pd.DataFrame()
        for group in train_group:
            corr_data[group[0]] = group[1][names[i]].copy().reset_index(drop=True)

        for group in test_group:
            corr_data[group[0]] = group[1][names[i]].copy().reset_index(drop=True)

        for group in capitals_group:
            corr_data[group[0]] = group[1][names[i]].copy().reset_index(drop=True)

        corr_data = corr_data.astype('float').corr()
        corr_data = corr_data.iloc[0:len(train_group),len(train_group):]
        im = axs[i].imshow(corr_data,cmap='Greys_r', vmin=-0.2, vmax=1)
        axs[i].axvline(x=len(test_group)-0.5, color='tab:blue', linestyle='-', linewidth=2)
        axs[i].set_xticks(ticks=np.arange(corr_data.shape[1]))
        axs[i].set_xticklabels(labels=(['']*corr_data.shape[1]))
        axs[i].set_yticks(ticks=np.arange(corr_data.shape[0]))
        axs[i].set_yticklabels(labels=(['']*corr_data.shape[0]))
        axs[i].set_title(name_dic[names[i]])
        axs[i].set_ylabel('Training')
        axs[i].text(6,len(train_group)+6,'Testing')
        axs[i].text(len(test_group)+6,len(train_group)+6,'Capital Cities')
    plt.colorbar(im, ax=axs.ravel().tolist(), shrink=0.8, pad=0.02)
    plt.show()    


if __name__ == '__main__':
    font={'size':16}
    mpl.rc('font',**font)

    samples_corr()
    score_avg_mos_corr()
    

