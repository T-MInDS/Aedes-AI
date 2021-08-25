import os, sys, pickle
import pandas as pd, numpy as np
from itertools import chain
from glob import glob
import math, pdb
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, rgb2hex, LogNorm
from matplotlib.colorbar import ColorbarBase
import os
os.environ["PROJ_LIB"] = "C:\\Utilities\\Python\\Anaconda\\Library\\share"; #fixr
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
sys.path.append("./utils")
from performance_metrics import *

def create_avg_mols():
    font={'size':30}
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
    ## https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html
    shp_info = m.readshapefile('./figures/cb_2018_us_state_20m','states',drawbounds=True,
                               linewidth=0.45,color='dimgray')

    data = pd.read_csv('./results/Capitals/Capitals_gru_hi_predictions.csv')
    data = data[~data.Location.str.contains('85')]
    data = data[~data.Location.str.contains('California')]
    data = data[~data.Location.str.contains('Arizona')]
    data = data[~data.Location.str.contains('Texas')]
    data = data[~data.Location.str.contains('Wisconsin')]
    data = data[~data.Location.str.contains('Minnesota')]
    data = data[~data.Location.str.contains('North Carolina')]
    data = data[~data.Location.str.contains('Delaware')]
    data = data[~data.Location.str.contains('New Jersey')]

    grouped = data.groupby(['Location'])
    
    vals = {group[1].Location.iloc[0].split(',')[-1]: np.mean(group[1]['MoLS']) for group in grouped}

    #%% -------- choose a color for each state based on score. -------
    colors={}
    statenames=[]
    cmap = plt.cm.viridis # use 'reversed hot' colormap
    vmin = min(vals.values()); vmax = max(vals.values()) # set range.
    norm = LogNorm(vmin=(vmin), vmax=(vmax))
    for shapedict in m.states_info:
        statename = shapedict['NAME']
        try:
            val = vals[statename]
            # calling colormap with value between 0 and 1 returns
            # rgba value.  Invert color range (hot colors are high
            # population), take sqrt root to spread out colors more.
            colors[statename] = cmap(norm(val))[:3]
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
    cb.set_ticks([2e1, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800])
    cb.set_ticklabels(['20', '', '', '', '', '', '', '', '100', '', '', '', '', '', '', '800'])
    
    #cb.set_ticklabels([str(norm(vmin)), '10e2', str(norm(vmax))])
    #cb.set_ticklabels(['10e5'])

    plt.show()
    return


def create_score_map():
    font={'size':16}
    mpl.rc('font',**font)
    
    fig, ax = plt.subplots()

    # Lambert Conformal map of lower 48 states.
    m = Basemap(llcrnrlon=-119,llcrnrlat=20,urcrnrlon=-64,urcrnrlat=49,
                projection='lcc',lat_1=33,lat_2=45,lon_0=-95)


    #%% ---------   draw state boundaries  ----------------------------------------
    ## data from U.S Census Bureau
    ## https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html
    
    shp_info = m.readshapefile('./figures/cb_2018_us_state_20m','states',drawbounds=True,
                               linewidth=0.45,color='dimgray')

    data = pd.read_csv('./results/Capitals/capitals_combined_scores.csv')

    data = data[data.State.str.contains('45')]
    data = data[~data.State.str.contains('California')]
    data = data[~data.State.str.contains('Arizona')]
    data = data[~data.State.str.contains('Texas')]
    data = data[~data.State.str.contains('Minnesota')]
    data = data[~data.State.str.contains('Wisconsin')]
    data = data[~data.State.str.contains('North Carolina')]
    data = data[~data.State.str.contains('Delaware')]
    data = data[~data.State.str.contains('New Jersey')]

    vals = {data.State.iloc[i].split(',')[-1]: data.Score.iloc[i] for i in range(0,len(data))}

    #%% -------- choose a color for each state based on score. -------
    colors={}
    statenames=[]
    cmap = plt.cm.viridis_r # use 'reversed hot' colormap
    vmin = min(data.Score); vmax = max(data.Score) # set range.
    norm = LogNorm(vmin=vmin, vmax=vmax)
    for shapedict in m.states_info:
        statename = shapedict['NAME']
        try:
            val = vals[statename]
            # calling colormap with value between 0 and 1 returns
            # rgba value.  Invert color range (hot colors are high
            # population), take sqrt root to spread out colors more.
            colors[statename] = cmap(norm(val))[:3]
        except:
            colors[statename] = 'gray'
        statenames.append(statename)

    #%% ---------  cycle through state names, color each one.  --------------------
    for nshape,seg in enumerate(m.states):
        # skip DC and Puerto Rico.
        if statenames[nshape] not in ['Puerto Rico']:
            color = rgb2hex(colors[statenames[nshape]])
            poly = Polygon(seg,facecolor=color,edgecolor=color)
            ax.add_patch(poly)


    #%% ---------   Show color bar  ---------------------------------------
    ax_c = fig.add_axes([0.8, 0.16, 0.03, 0.7])
    cb = ColorbarBase(ax_c,cmap=cmap,norm=norm,orientation='vertical')
    cb.set_ticks([0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 1.0])
    cb.set_ticklabels(['0.2', '', '0.4', '0.6', '', '0.8', '1.0'])

    plt.show()
    return

def create_heatmap():
        
    fig, ax = plt.subplots()

    im = plt.imread('./figures/capital_map.png')
    ax.imshow(im)
    ax.axis('off')

    im=plt.imread('./figures/avg_mols.png')
    newax = fig.add_axes([0.04,0.08,0.45,0.22])
    newax.imshow(im)
    newax.axis('off')

    plt.show()
    
if __name__ == '__main__':
    create_avg_mols()
    create_score_map()
    create_heatmap()
    
