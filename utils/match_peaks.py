import numpy as np, pandas as pd
import os, argparse, json, pdb
from glob import glob
import matplotlib.pyplot as plt
from collections import defaultdict

#python match_peaks.py -r results/Test/Test_ff_model_predictions

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--results', type = str, default = '', help = 'The file for results that have already been generated.')
    parser.add_argument('-c', '--county', type = str, default = '', help = 'The specific county, state, and year to visualize point matchings for.')
    return parser.parse_args()

def load_results_data(filename):
    results = pd.read_csv(filename)
    groups = results.groupby(by = 'Location')
    output = {}
    for citystate, subset in groups:
        city, state = citystate.split(',')
        for year in range(2011, 2021):
            y_pred = subset[subset['Year'] == year]['Neural Network']
            y_true = subset[subset['Year'] == year]['MoLS']
            y_pred, y_true = y_pred / y_true.max(), y_true / y_true.max() 
            output[(city, state, str(year))] = (y_pred.values, y_true.values)
    return output

def peak_finder(array, threshold = 0.8, on_confidence = 7, off_confidence = 0):
    peaks = []
    above_threshold = array > threshold
    peak_start = 0 
    peaking = False
    for i, h in enumerate(above_threshold):
        if h and not peaking:
            #meet peak on condition
            if above_threshold[i:i + on_confidence].all():
                peak_start = i
                peaking = True
        elif not h and peaking:
            #meet peak off condition
            if (~above_threshold[i:i + off_confidence]).all():
                peaks.append((peak_start, i))
                peaking = False
    if peaking:
        peaks.append((peak_start, i))
    return peaks

def season_length(array, threshold = 0.2, on_confidence = 7, off_confidence = 0):
    peaks = peak_finder(array, threshold, on_confidence, off_confidence)
    if peaks:
        ons, offs = zip(*peaks)
        return max(offs) - min(ons)
    return 364

def match_peaks(offsets, num_fake, num_real):
    on_order = sorted(offsets, key = lambda x: abs(offsets[x][0]))
    off_order = sorted(offsets, key = lambda x: abs(offsets[x][1]))
    on_matching, off_matching = {}, {}
    on_chosen, off_chosen = [], []
    for i in range(len(offsets)):
        real, fake = on_order[i]
        if real not in on_matching.keys():
            if fake not in on_matching.values():
                on_matching[real] = fake
                on_chosen.append(offsets[(real, fake)][0])
        real, fake = off_order[i]
        if real not in off_matching.keys():
            if fake not in off_matching.values():
                off_matching[real] = fake
                off_chosen.append(offsets[(real, fake)][1])
    return on_chosen, off_chosen, on_matching, off_matching

def min_offset(peaks_fake, peaks_real):
    num_peaks_fake = len(peaks_fake)
    num_peaks_real = len(peaks_real)
    #calculate offsets
    offsets = {}
    for i, real in enumerate(peaks_real):
        for j, fake in enumerate(peaks_fake):
            offset = (real[0] - fake[0], real[1] - fake[1])
            offsets[(i, j)] = offset
    if not offsets:
        return {'D_on': [np.nan], 'D_off': [np.nan], 'D_pairs': []}
    d_on, d_off, m_on, m_off = match_peaks(offsets, num_peaks_fake, num_peaks_real)
    point_pairs = []
    for real, fake in m_on.items():
        point_pairs.append((peaks_real[real][0], peaks_fake[fake][0]))
    for real, fake in m_off.items():
        point_pairs.append((peaks_real[real][1], peaks_fake[fake][1]))
    return {'D_on': d_on, 'D_off': d_off, 'D_pairs': point_pairs}

def compare_peaks(output, metric, threshold=0.8,
                        on_confidence = 7, off_confidence = 0):
    results = {}
    for citystateyear, (y_pred, y_true) in output.items():
        peaks_pred = peak_finder(y_pred, threshold=threshold, on_confidence=on_confidence, off_confidence=off_confidence)
        peaks_true = peak_finder(y_true, threshold=threshold, on_confidence=on_confidence, off_confidence=off_confidence)
        results[citystateyear] = metric(peaks_pred, peaks_true)
    return results

def to_latex(on_table, off_table, model_name):
    model_name = model_name.replace('_', ' ').upper()
    formatter = lambda d: str(round(d[0], 3)) + ' $\\pm$ ' + str(round(d[1], 3))
    rstring = '\\multirow{2}{*}{' + model_name + '} & Mean $D_{on}$ & '
    rstring += ' & '.join(map(formatter, zip(on_table.mean()[1:], on_table.std()[1:])))
    rstring += ' \\\\\n & Mean $D_{off}$ & '
    rstring += ' & '.join(map(formatter, zip(off_table.mean()[1:], off_table.std()[1:])))
    return rstring + '\\\\\n\\hline\n'
    
def main():
    args = parse_args()
    county = tuple(args.county.split(','))

    output = load_results_data(args.results)

    columns = ['SeasonWidth', '20%', '40%', '60%', '80%']
    on_table = pd.DataFrame(columns = columns, index = pd.MultiIndex.from_tuples(output.keys(), names = ['City', 'State', 'Year']))
    off_table = pd.DataFrame(columns = columns, index = pd.MultiIndex.from_tuples(output.keys(), names = ['City', 'State', 'Year']))

    season_width = defaultdict(list)
    for (city, state, _), (_, y_true) in output.items():
        season_width[(city, state)].append(season_length(y_true, 0.2, 7, 7))    
    season_width = {key: np.mean(widths) for key, widths in season_width.items()}

    point_pairs = []

    for threshold in [0.2, 0.4, 0.6, 0.8]:
        threshold_str = str(int(100 * threshold)) + '%'
        
        results = compare_peaks(output, min_offset, threshold, 7, 7)
        for index, result in results.items():
            # index is (city, state, year)
            citystate = index[:-1]
            
            if index == county:
                point_pairs.extend(result['D_pairs'])

            on_table.loc[index, 'SeasonWidth'] = season_width[citystate]
            on_table.loc[index, threshold_str] = np.mean(result['D_on']) / season_width[citystate]
            
            off_table.loc[index, 'SeasonWidth'] = season_width[citystate]
            off_table.loc[index, threshold_str] = np.mean(result['D_off']) / season_width[citystate]

            # SeasonLength = f(c, s)
            # D_csy = f(c, s, y) / SeasonLength(c, s)
            
            # Mean = mean(D for c, for s, for y)

<<<<<<< HEAD
    if False:
        terms = args.results.split('_')
        modname = '_'.join([terms[1]] + terms[3:-1])
        directory = args.results.split('/')[1]
=======
    terms = args.results.split('_')
    #modname = '_'.join([terms[1]] + terms[3:-1])
    directory = args.results.split('/')[-1].split('_')[0]
    modname=args.results.split(directory+'_')[-1].split('_predictions')[0]
>>>>>>> revisions

    if args.county:
        fake, real = output[county]
        plt.plot(real, marker = '', label = 'MoLS')
        plt.plot(fake, marker = '', label = modname)
        for rp, fp in point_pairs:
            x, y = [rp, fp], [real[rp], fake[fp]]
            plt.plot(x, y, marker = 'o', color = 'black')
        plt.show()

<<<<<<< HEAD
    on_table.to_csv('D_on_table.csv')
    off_table.to_csv('D_off_table.csv')
    #with open('./results/Tables/'+ directory + '/' + modname + '_latex.txt', 'w') as fp:
    #    fp.write(to_latex(on_table, off_table, modname))
=======
    print(modname)
    on_table.to_csv('./results/Threshold_tables/'+ directory + '/' + modname + '_D_on_table.csv')
    off_table.to_csv('./results/Threshold_tables/'+ directory + '/' + modname + '_D_off_table.csv')
    with open('./results/Threshold_tables/'+ directory + '/' + modname + '_latex.txt', 'w') as fp:
        fp.write(to_latex(on_table, off_table, modname))
>>>>>>> revisions

if __name__ == '__main__':
    main()
