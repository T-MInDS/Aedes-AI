import os, sys
import pandas as pd, numpy as np
from itertools import chain
from glob import glob
import math, pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
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
    shp_info_ = m_.readshapefile('st99_d00','states',drawbounds=False)

    ## population density by state from
    ## http://en.wikipedia.org/wiki/List_of_U.S._states_by_population_density
    popdensity = {
    'New Jersey':  438.00,
    'Rhode Island':   387.35,
    'Massachusetts':   312.68,
    'Connecticut':	  271.40,
    'Maryland':   209.23,
    'New York':    155.18,
    'Delaware':    154.87,
    'Florida':     114.43,
    'Ohio':	 107.05,
    'Pennsylvania':	 105.80,
    'Illinois':    86.27,
    'California':  83.85,
    'Hawaii':  72.83,
    'Virginia':    69.03,
    'Michigan':    67.55,
    'Indiana':    65.46,
    'North Carolina':  63.80,
    'Georgia':     54.59,
    'Tennessee':   53.29,
    'New Hampshire':   53.20,
    'South Carolina':  51.45,
    'Louisiana':   39.61,
    'Kentucky':   39.28,
    'Wisconsin':  38.13,
    'Washington':  34.20,
    'Alabama':     33.84,
    'Missouri':    31.36,
    'Texas':   30.75,
    'West Virginia':   29.00,
    'Vermont':     25.41,
    'Minnesota':  23.86,
    'Mississippi':	 23.42,
    'Iowa':	 20.22,
    'Arkansas':    19.82,
    'Oklahoma':    19.40,
    'Arizona':     17.43,
    'Colorado':    16.01,
    'Maine':  15.95,
    'Oregon':  13.76,
    'Kansas':  12.69,
    'Utah':	 10.50,
    'Nebraska':    8.60,
    'Nevada':  7.03,
    'Idaho':   6.04,
    'New Mexico':  5.79,
    'South Dakota':	 3.84,
    'North Dakota':	 3.59,
    'Montana':     2.39,
    'Wyoming':      1.96,
    'Alaska':     0.42}

    #%% -------- choose a color for each state based on population density. -------
    colors={}
    statenames=[]
    cmap = plt.cm.hot_r # use 'reversed hot' colormap
    vmin = 0; vmax = 450 # set range.
    norm = Normalize(vmin=vmin, vmax=vmax)
    for shapedict in m.states_info:
        statename = shapedict['NAME']
        # skip DC and Puerto Rico.
        if statename not in ['District of Columbia','Puerto Rico']:
            pop = popdensity[statename]
            # calling colormap with value between 0 and 1 returns
            # rgba value.  Invert color range (hot colors are high
            # population), take sqrt root to spread out colors more.
            colors[statename] = cmap(np.sqrt((pop-vmin)/(vmax-vmin)))[:3]
        statenames.append(statename)

    #%% ---------  cycle through state names, color each one.  --------------------
    for nshape,seg in enumerate(m.states):
        # skip DC and Puerto Rico.
        if statenames[nshape] not in ['Puerto Rico', 'District of Columbia']:
            color = rgb2hex(colors[statenames[nshape]])
            poly = Polygon(seg,facecolor=color,edgecolor=color)
            ax.add_patch(poly)

    AREA_1 = 0.005  # exclude small Hawaiian islands that are smaller than AREA_1
    AREA_2 = AREA_1 * 30.0  # exclude Alaskan islands that are smaller than AREA_2
    AK_SCALE = 0.19  # scale down Alaska to show as a map inset
    HI_OFFSET_X = -1900000  # X coordinate offset amount to move Hawaii "beneath" Texas
    HI_OFFSET_Y = 250000    # similar to above: Y offset for Hawaii
    AK_OFFSET_X = -250000   # X offset for Alaska (These four values are obtained
    AK_OFFSET_Y = -750000   # via manual trial and error, thus changing them is not recommended.)

    for nshape, shapedict in enumerate(m_.states_info):  # plot Alaska and Hawaii as map insets
        if shapedict['NAME'] in ['Alaska', 'Hawaii']:
            seg = m_.states[int(shapedict['SHAPENUM'] - 1)]
            if shapedict['NAME'] == 'Hawaii' and float(shapedict['AREA']) > AREA_1:
                seg = [(x + HI_OFFSET_X, y + HI_OFFSET_Y) for x, y in seg]
                color = rgb2hex(colors[statenames[nshape]])
            elif shapedict['NAME'] == 'Alaska' and float(shapedict['AREA']) > AREA_2:
                seg = [(x*AK_SCALE + AK_OFFSET_X, y*AK_SCALE + AK_OFFSET_Y)\
                       for x, y in seg]
                color = rgb2hex(colors[statenames[nshape]])
            poly = Polygon(seg, facecolor=color, edgecolor='gray', linewidth=.45)
            ax.add_patch(poly)

    ax.set_title('United states population density by state')

        #%% ---------   Show color bar  ---------------------------------------
    ax_c = fig.add_axes([0.9, 0.1, 0.03, 0.8])
    cb = ColorbarBase(ax_c,cmap=cmap,norm=norm,orientation='vertical',
                      label=r'[population per $\mathregular{km^2}$]')


    pdb.set_trace()
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
