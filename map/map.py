import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, pdb
os.environ["PROJ_LIB"] = "C:\\Utilities\\Python\\Anaconda\\Library\\share"; #fixr
from mpl_toolkits.basemap import Basemap as Basemap


if __name__ == '__main__':

    fig, ax = plt.subplots()

    # US: llcrnrlon=-130, llcrnrlat=25, urcrnrlon=-65.,urcrnrlat=52.
    # SW: llcrnrlon=-125, llcrnrlat=30, urcrnrlon=-108.,urcrnrlat=43.
    # FL: llcrnrlon=-90, llcrnrlat=24, urcrnrlon=-78.,urcrnrlat=32.
    # NE: llcrnrlon=-82, llcrnrlat=39, urcrnrlon=-70.,urcrnrlat=43.
    m = Basemap(llcrnrlon=-130, llcrnrlat=25, urcrnrlon=-65.,urcrnrlat=52.,
                resolution='i', lat_0 = 40., lon_0 = -80)


    # From US Census Bureau
    # https://www.census.gov/geographies/mapping-files/time-series/geo/carto-boundary-file.html
    m.readshapefile('./map/cb_2018_us_state_20m','states',drawbounds=True,
                            linewidth=0.75,color='gray')
    
    m.drawcounties(color='gray')


    data=pd.read_csv("./map/Aedes_counties.csv")
    # Training Counties
    train_lat=data[data.Boolean_Training==0]["Latitude"].values
    train_lon=data[data.Boolean_Training==0]["Longitude"].values
    # Testing Counties
    test_lat=data[data.Boolean_Training==1]["Latitude"].values
    test_lon=data[data.Boolean_Training==1]["Longitude"].values
    # Capital Cities
    cap_lat=data[data.Boolean_Training==2]["Latitude"].values
    cap_lon=data[data.Boolean_Training==2]["Longitude"].values

    c1=[0,158/255,115/255]
    c2=[230/255,159/255,0]

    m.scatter(train_lon,train_lat,latlon=True,c=[(c1)],s=10*np.ones(len(train_lat)),marker='s')
    m.scatter(test_lon,test_lat,latlon=True,c=[(c2)],s=15*np.ones(len(test_lat)),marker='v')
    m.scatter(cap_lon,cap_lat,latlon=True,c='red',s=15*np.ones(len(cap_lat)),marker='*')
    plt.show()
