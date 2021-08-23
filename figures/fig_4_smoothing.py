import numpy as np, pandas as pd
import os, argparse, json, sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
sys.path.append("./utils")
from match_peaks import compare_peaks, min_offset


def main():
    #font={'size':16}
    #mpl.rc('font',**font)

    fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True, sharex=True)
    models=['ff', 'lstm', 'gru']

    for i in range(0,len(models)):    
        filename='./results/Test/Test_{}_predictions.csv'.format(models[i])
        raw_name='./results/Test/Test_{}_raw_predictions.csv'.format(models[i])
        data=pd.read_csv(filename)
        data=data[(data.Location=='Avondale,Arizona') & (data.Year==2020)]

        raw_data=pd.read_csv(raw_name)
        raw_data=raw_data[(raw_data.Location=='Avondale,Arizona') & (raw_data.Year==2020)]

        axs[i].plot(np.arange(len(data.MoLS)), data['Neural Network'], label='Smoothed', color='k')
        axs[i].plot(np.arange(len(data.MoLS)), raw_data['Neural Network'], label='Raw', color='r', linestyle='--')
        axs[i].set_xlabel('Days since 01/01/2020')
        axs[i].set_title(models[i].upper())
        if i<1:
            axs[i].set_ylabel('Abundance')
            axs[i].legend(loc='upper left')
    
    #axs[1].plot(np.arange(len(data.MoLS)), data.MoLS, label='MoLS', color='k')
    #axs[1].plot(np.arange(len(data.MoLS)), data['Neural Network'], label='Smoothed GRU', color='r', linestyle='--')
    #axs[1].set_xlabel('Days since 01/01/2020')
    #axs[1].legend(loc='upper right')

    plt.show()
    fig.savefig("raw_smoothed.png", dpi=300)


if __name__ == '__main__':
    main()
