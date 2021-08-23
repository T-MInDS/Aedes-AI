import numpy as np, pandas as pd
import os, argparse, json, sys
import matplotlib.pyplot as plt
sys.path.append("./utils")
from collections import defaultdict
from match_peaks import compare_peaks, min_offset

def load_results_data(filename,year):
    results = pd.read_csv(filename)
    groups = results.groupby(by = 'Location')
    output = {}
    for citystate, subset in groups:
        city, state = citystate.split(',')
        # 2011 starts mid-year so only include 2012-2020
        y_pred = subset[subset['Year'] != 2011]['Neural Network']
        y_true = subset[subset['Year'] != 2011]['MoLS']
        output[(city, state, str(year))] = (y_pred.values, y_true.values)
    return output

def main():
    filename='./results/Test/Test_gru_hi_predictions.csv'
    modname = filename.split("/Test")[-1].split('_predictions')[0]
    modname = modname.replace('_',' ').upper()
    #modname = 'LSTM'
    ct=('Waukesha,Wisconsin', 'Riverside,California', 'Ventura,California', 'Shasta,California', 'Yuba,California', 'Butte,California', 'Stanislaus,California', 'Yolo,California', 'Colusa,California',
        'Fortuna Foothills,Arizona', 'Prescott Valley,Arizona', 'Maricopa,Arizona', 'Nogales,Arizona', 'Eloy,Arizona', 'Marana,Arizona', 'Tucson,Arizona', 'Avondale,Arizona',
        'Fairfield,Connecticut', 'Forsyth,North Carolina', 'Pitt,North Carolina', 'New York,New York', 'Richmond,New York', 'Salem,New Jersey', 'Essex,New Jersey', 'Cameron,Texas',
        'Osceola,Florida', 'Pinellas,Florida', 'Collier,Florida', 'Jackson,Florida')
        #'Fortuna Foothills,Arizona', 'Maricopa,Arizona', 'Eloy,Arizona', 'Avondale,Arizona', 'Marana,Arizona', 'Prescott Valley,Arizona', 'Tucson,Arizona', 'Nogales,Arizona', 'Waukesha,Wisconsin', 'Cameron,Texas', 'Ventura,California', 'Riverside,California', 'Shasta,California', 'Colusa,California', 'Yuba,California', 'Yolo,California', 'Butte,California', 'Stanislaus,California', 'Richmond,New York', 'New York,New York', 'Fairfield,Connecticut', 'Forsyth,North Carolina', 'Pitt,North Carolina', 'Salem,New Jersey', 'Essex,New Jersey', 'Jackson,Florida', 'Osceola,Florida', 'Collier,Florida', 'Pinellas,Florida')
    idx=0
    figidx=1
    fig, axs = plt.subplots(3, 2, sharex=True,figsize=(15, 8))
    output = load_results_data(filename,2012)
    for cty in ct:
        print(cty)
        ct1=(cty.split(',')[0],cty.split(',')[1],'2012')
        fake, real = output[ct1]
        axs[idx // 2,idx % 2].plot(real, 'k-', label = 'MoLS')
        axs[idx // 2,idx % 2].plot(fake, 'r--', label = modname)
        if idx == 0:
            axs[idx // 2,idx % 2].legend(loc='upper right',ncol=2)
        if cty.split(',')[1] != 'Arizona':
            axs[idx // 2,idx % 2].set_title(ct1[0]+" County, "+ct1[1])
        else:
            axs[idx // 2,idx % 2].set_title(ct1[0]+", "+ct1[1])
        if idx // 2 == 2:
            axs[idx // 2,idx % 2].set_xlabel("Days since 01/01/2012")
        idx=idx+1
        if idx % 6 == 0:
            fig.savefig("GRU_HI_Plots_by_Counties_"+str(figidx)+".png", dpi=300, bbox_inches='tight')
            idx=0
            figidx=figidx+1
            fig, axs = plt.subplots(3, 2, sharex=True, figsize=(15, 8))
    axs[2,1].set_frame_on(0)
    axs[2,1].set_xticks(())
    axs[2,1].set_yticks(())
    fig.savefig("GRU_HI_Plots_by_Counties_"+str(figidx)+".png", dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()
