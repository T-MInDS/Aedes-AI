import numpy as np, pandas as pd
import os, argparse, json, sys, pdb
import matplotlib.pyplot as plt
sys.path.append("./utils")
from collections import defaultdict
from match_peaks import compare_peaks, min_offset

def load_results_data(filename,year):
    results = pd.read_csv(filename)
    groups = results.groupby(by = 'Location')
    output = {}
    for state, subset in groups:
        # 2011 starts mid-year so only include 2012-2020
        y_pred = subset[subset['Year'] != 2011]['Neural Network']
        y_true = subset[subset['Year'] != 2011]['MoLS']
        output[(state.split(',')[-1], str(year))] = (y_pred.values, y_true.values)
    return output

def main():
    filename='./results/Capitals/Capitals_gru_hi_predictions.csv'
    modname = filename.split("/Capitals")[-1].split('_predictions')[0]
    modname = modname.replace('_',' ').upper()
    #modname = 'LSTM'
    ct=('Montana','Nevada','Michigan','Kentucky','Georgia','Louisiana',)
    idx=0
    figidx=1
    fig, axs = plt.subplots(3, 2, sharex=True,figsize=(15, 8))
    output = load_results_data(filename,2012)
    for cty in ct:
        print(cty)
        ct1=(cty,'2012')
        fake, real = output[ct1]
        axs[idx // 2,idx % 2].plot(real, 'k-', label = 'MoLS')
        axs[idx // 2,idx % 2].plot(fake, 'r--', label = modname)
        if idx == 0:
            axs[idx // 2,idx % 2].legend(loc='upper left',ncol=2)
        axs[idx // 2,idx % 2].set_title(ct1[0])
        if idx // 2 == 2:
            axs[idx // 2,idx % 2].set_xlabel("Days since 01/01/2012")
        idx=idx+1
    #axs[2,1].set_frame_on(0)
    #axs[2,1].set_xticks(())
    #axs[2,1].set_yticks(())
    plt.show()
    fig.savefig("Capitals_GRU_HI_Plots_"+str(figidx)+".png", dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()
