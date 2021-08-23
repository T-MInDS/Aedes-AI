import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import pdb

HI_LO_BASES = ['lr', 'lr_hi', 'lr_lo', 'lr_hi_lo',
               'ff', 'ff_hi', 'ff_lo', 'ff_hi_lo',
               'gru', 'gru_hi', 'gru_lo', 'gru_hi_lo',
               'lstm', 'lstm_hi', 'lstm_lo', 'lstm_hi_lo']

VARIANT_BASES = ['ff', 'ff\'', 'ff_2000', 'ff_120',
               'lstm', 'lstm\'', 'lstm_2000', 'lstm_120',
               'gru', 'gru\'', 'gru_2000', 'gru_120']
TABLE_BASES = [*HI_LO_BASES, *VARIANT_BASES]

def build_table(table_bases, metric, mode):
    columns = ["Model", "20%", "40%", "60%", "80%"]
    data = []
    for base in table_bases:
        fname = './results/Threshold_tables/Test/' + base + '_' + metric + '_table.csv'
        tab = pd.read_csv(fname, header = 0)
        #tab = tab[(tab.State=='Florida') & (tab.City=='Collier') & (tab.Year==2020)]
        #tab = tab[tab.State=='Florida']
        add_to_data = [base]
        for threshold in ["20%", "40%", "60%", "80%"]:
            if mode == 'mean':
                value = tab[threshold].mean()
            elif mode == 'std':
                value = tab[threshold].std()
            elif mode == 'p_nan':
                value = tab[threshold].isna().sum() / len(tab)
            add_to_data.append(round(value, 3))
        data.append(add_to_data)
    return pd.DataFrame(data, columns = columns)

def merge_tables(d_on, d_off):
    columns = ['Model', 'Metric', "20%", "40%", "60%", "80%"]
    metric_map = {"D_on": d_on, "D_off": d_off}
    data = []
    for i in range(len(d_on)):
        for metric, tab in metric_map.items():
            add_to_data = []
            add_to_data.append(tab.iloc[i]['Model'])
            add_to_data.append(metric)
            add_to_data.extend(tab.iloc[i].loc["20%":].values)
            data.append(add_to_data)
    return pd.DataFrame(data, columns = columns)

def rank_models(mean_tab, std_tab):
    results = defaultdict(list)
    for threshold in ("20%", "40%", "60%", "80%"):
        means = mean_tab[threshold].abs()
        stds = std_tab[threshold]
        ranking = (means * stds).argsort()
        for i in range(len(ranking)):
            results[threshold].append(mean_tab["Model"].iloc[ranking.iloc[i]])
    return results

def reverse_rank(rank):
    results = defaultdict(list)
    for threshold in ("20%", "40%", "60%", "80%"):
        for i in range(len(rank[threshold])):
            results[rank[threshold][i]].append(12 - i)
    return results

def plot_rank(rank, metric):
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20.colors)
    for model in sorted(rank):
        plt.plot([20, 40, 60, 80], rank[model], label = model)
    plt.title("Model Ranks for " + metric + " Metric")
    plt.yticks(list(range(1, 13))[::1], list(range(1, 13))[::-1])
    plt.xticks([20, 40, 60, 80])
    plt.legend()
    plt.show()

def main():
    mean_tab_d_on = build_table(TABLE_BASES, 'D_on', 'mean')
    std_tab_d_on = build_table(TABLE_BASES, 'D_on', 'std')
    nan_tab_d_on = build_table(TABLE_BASES, 'D_on', 'p_nan')
    mean_tab_d_off = build_table(TABLE_BASES, 'D_off', 'mean')
    std_tab_d_off = build_table(TABLE_BASES, 'D_off', 'std')
    nan_tab_d_off = build_table(TABLE_BASES, 'D_off', 'p_nan')

    mean_tab = merge_tables(mean_tab_d_on, mean_tab_d_off)
    std_tab = merge_tables(std_tab_d_on, std_tab_d_off)
    nan_tab = merge_tables(nan_tab_d_on, nan_tab_d_off)

    mean_tab.to_csv('./results/Threshold_tables/Test/all_mean.csv')
    std_tab.to_csv('./results/Threshold_tables/Test/all_std.csv')
    nan_tab.to_csv('./results/Threshold_tables/Test/all_nan.csv')


    on_rank = rank_models(mean_tab_d_on, std_tab_d_on)
    off_rank = rank_models(mean_tab_d_off, std_tab_d_off)

    on_rev, off_rev = reverse_rank(on_rank), reverse_rank(off_rank)

    #plot_rank(on_rev, "$D_{on}$")
    #plot_rank(off_rev, "$D_{off}$")

    with open('./results/Threshold_tables/Test/all_latex.txt', 'w') as fp:
        for base in TABLE_BASES:
            fname = './results/Threshold_tables/Test/' + base + '_latex.txt'
            with open(fname) as rf:
                fp.write(rf.read())

if __name__ =='__main__':
    main()
