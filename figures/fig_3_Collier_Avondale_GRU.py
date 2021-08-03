import numpy as np, pandas as pd
import os, argparse, json, sys
import matplotlib.pyplot as plt
from collections import defaultdict
sys.path.append("./utils")
from match_peaks import compare_peaks, min_offset

def load_results_data(filename,year):
    results = pd.read_csv(filename)
    groups = results.groupby(by = 'Location')
    output = {}
    for citystate, subset in groups:
        city, state = citystate.split(',')
        y_pred = subset[subset['Year'] == year]['Neural Network']
        y_true = subset[subset['Year'] == year]['MoLS']
        y_pred, y_true = y_pred / y_true.max(), y_true / y_true.max() 
        output[(city, state, str(year))] = (y_pred.values, y_true.values)
    return output

def county_data(filename,county):
    output = load_results_data(filename,2020)
    point_pairs = []
    for threshold in [0.2, 0.4, 0.6, 0.8]:       
        results = compare_peaks(output, min_offset, threshold, 7, 7)
        for index, result in results.items():
            citystate = index[:-1]            
            if index == county:
                point_pairs.extend(result['D_pairs'])
    fake, real = output[county]
    return fake, real, point_pairs

def plot_thresholds(filename,county,modname,ax):
    fake, real, point_pairs=county_data(filename,county)
    ax.plot(real, 'k-', label = 'MoLS')
    ax.plot(fake, 'r--', label = modname)
    for rp, fp in point_pairs:
        x, y = rp, real[rp]
        ax.scatter(x, y, marker = 'o', color = 'black')
        x, y = fp, fake[fp]
        ax.scatter(x, y, marker = 'o', color = 'red')
        x, y = [rp, fp], [real[rp], fake[fp]]
        ax.plot(x, y, 'k:')
    ax.set_xlabel("Days since 01/01/" + county[2])
    ax.legend()

def main():
    filename='./results/Test/Test_gru_predictions.csv'
    ct1=('Collier','Florida','2020')
    ct2=('Avondale','Arizona','2020')
    # modname = filename.split("_")[1]
    modname = 'GRU'
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    plot_thresholds(filename,ct1,modname,axs[0])
    axs[0].set_ylabel('Scaled Incidence')
    axs[0].set_title(ct1[0]+" County, "+ct1[1])
    plot_thresholds(filename,ct2,modname,axs[1])
    plt.title(ct2[0]+", "+ct2[1])
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    #fig1.savefig("Collier-Avondale_"+modname+".png", dpi=300)

if __name__ == '__main__':
    main()
