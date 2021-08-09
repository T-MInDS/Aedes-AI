import os, sys
import pandas as pd, numpy as np
from itertools import chain
from glob import glob
import math, pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, rgb2hex
from matplotlib.colorbar import ColorbarBase
import os
os.environ["PROJ_LIB"] = "C:\\Utilities\\Python\\Anaconda\\Library\\share"; #fixr
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
sys.path.append("./utils")
from performance_metrics import *


def sort_state(data,state):
    subset=data.filter(regex=state)
    subset=(subset.mean(axis=0)).sort_values(axis=0,ascending=False)
    return subset.index

def sort_models(data):
    data=(data.mean(axis=1)).sort_values(axis=0,ascending=True)
    return data.index

def create_heatmap(heatmap, states, x_labels, y_labels):
    font={'size':16}
    mpl.rc('font',**font)
    
    fig, ax = plt.subplots()

    # Lambert Conformal map of lower 48 states.
    m = Basemap(llcrnrlon=-119,llcrnrlat=20,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95)

    # Mercator projection, for Alaska and Hawaii
    m_ = Basemap(llcrnrlon=-190,llcrnrlat=20,urcrnrlon=-143,urcrnrlat=46,
                projection='merc',lat_ts=20)  # do not change these numbers

    #%% ---------   draw state boundaries  ----------------------------------------
    ## data from U.S Census Bureau
    ## http://www.census.gov/geo/www/cob/st2000.html
    shp_info = m.readshapefile('./figures/st99_d00','states',drawbounds=True,
                               linewidth=0.45,color='gray')
    shp_info_ = m_.readshapefile('./figures/st99_d00','states',drawbounds=False)

    data = pd.read_csv('./results/Capitals/capitals_map.csv')

    vals = {data.State.iloc[i]: data.Value.iloc[i] for i in range(0,len(data))}

    #%% -------- choose a color for each state based on score. -------
    colors={}
    statenames=[]
    cmap = plt.cm.viridis_r # use 'reversed hot' colormap
    vmin = min(data.Value); vmax = max(data.Value) # set range.
    norm = Normalize(vmin=vmin, vmax=vmax)
    for shapedict in m.states_info:
        statename = shapedict['NAME']
        try:
            val = vals[statename]
            # calling colormap with value between 0 and 1 returns
            # rgba value.  Invert color range (hot colors are high
            # population), take sqrt root to spread out colors more.
            colors[statename] = cmap(np.sqrt((val-vmin)/(vmax-vmin)))[:3]
        except:
            colors[statename] = 'gray'
        statenames.append(statename)

    #%% ---------  cycle through state names, color each one.  --------------------
    for nshape,seg in enumerate(m.states):
        # skip DC and Puerto Rico.
        if statenames[nshape] not in ['Puerto Rico', 'District of Columbia']:
            color = rgb2hex(colors[statenames[nshape]])
            poly = Polygon(seg,facecolor=color,edgecolor=color)
            ax.add_patch(poly)


    #%% ---------   Show color bar  ---------------------------------------
    ax_c = fig.add_axes([0.8, 0.16, 0.03, 0.7])
    cb = ColorbarBase(ax_c,cmap=cmap,norm=norm,orientation='vertical')

    plt.show()
    
def combined_metric():
    ddir='./results/'
    global_fit=pd.read_csv(ddir+"Capitals/capitals_scores.csv")
    locs=np.unique(global_fit.Location)
    thres_mo_off=pd.read_csv(ddir+"Threshold_tables/Capitals/gru_hi_D_off_table.csv")
    thres_mo_on=pd.read_csv(ddir+"Threshold_tables/Capitals/gru_hi_D_on_table.csv")

    ds=np.zeros((len(locs)))
    hs=np.zeros((len(locs)))
    for i in range(0, len(locs)):
        subset=global_fit[global_fit.Location==locs[i]]
        m1=(1-np.sqrt(subset.R2**2*subset.Pearson**2))
        m2=np.sqrt(subset.AUC_Diff**2+subset.RMSE**2)
        d=np.average(np.sqrt(m1**2+m2**2))
        ds[i]=d

        thres_off=thres_mo_off[(thres_mo_off.City.astype('str')==locs[i].split(',')[0]) & (thres_mo_off.State==locs[i].split(',')[-1])]
        thres_on=thres_mo_on[(thres_mo_on.City.astype('str')==locs[i].split(',')[0]) & (thres_mo_on.State==locs[i].split(',')[-1])]
        Z=(1+(thres_on[["20%","40%","60%","80%"]].isna()*1).sum(axis=0)/len(thres_on))
        for col in ["20%","40%","60%","80%"]:
            if thres_on[col].isna().any():
                thres_on[col].fillna(np.mean(np.absolute(thres_mo_on[col])), inplace=True)
            if thres_off[col].isna().any():
                thres_off[col].fillna(np.mean(np.absolute(thres_mo_off[col])), inplace=True)
            ht=1/sum(Z)*sum(Z*np.sqrt(np.mean(np.absolute(thres_on[["20%","40%","60%","80%"]]))*
                            np.std(np.absolute(thres_on[["20%","40%","60%","80%"]]))+
                            np.mean(np.absolute(thres_off[["20%","40%","60%","80%"]]))*
                            np.std(np.absolute(thres_off[["20%","40%","60%","80%"]]))))
            if math.isnan(ht):
                print(locs[i])
            hs[i]=ht
    ds=ds/np.nanmax(ds)
    hs=hs/np.nanmax(hs)
    S=np.sqrt(ds**2+hs**2)
    S=pd.DataFrame(S,index=locs)
    #S.to_csv(ddir+"Capitals/combined_scores.csv")
    return 
if __name__ == '__main__':

    scores=combined_metric()
    states=["Wisconsin","California","Arizona","Connecticut","North Carolina",
            "New York","New Jersey","Texas","Florida"]
    #loc_order=list()
    #for state in states:
    #    order=sort_state(scores,state)
    #    loc_order.append(order)
    #loc_order=list(chain(*loc_order))
    #mo_order=sort_models(scores)
    #scores=scores[loc_order].loc[mo_order]
    loc_order, mo_order=list(),list()
    create_heatmap(scores, states, loc_order, mo_order)
