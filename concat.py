import os, json, sys
import pandas as pd, numpy as np
import pdb
from glob import glob

def state_dic(state):
    dic={"AL": "Alabama", "AR": "Arkansas", "AZ": "Arizona", "CA": "California",
         "CO": "Colorado", "CT": "Connecticut", "DC": "District of Columbia",
         "DE": "Delaware", "FL": "Florida", "GA": "Georgia", "IA": "Iowa",
         "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "KS": "Kansas",
         "KY": "Kentucky", "LA": "Louisiana", "MA": "Massachusetts",
         "MD": "Maryland", "ME": "Maine", "MI": "Michigan", "MN": "Minnesota",
         "MO": "Missouri", "MS": "Mississippi", "MT": "Montana",
         "NC": "North Carolina", "ND": "North Dakota", "NE": "Nebraska",
         "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
         "NV": "Nevada", "NY": "New York", "OH": "Ohio", "OK": "Oklahoma",
         "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island",
         "SC": "South Carolina", "SD": "South Dakota", "TN": "Tennessee",
         "TX": "Texas", "UT": "Utah", "VA": "Virginia", "VT": "Vermont",
         "WA": "Washington", "WI": "Wisconsin", "WV": "West Virginia",
         "WY": "Wyoming"}
    return dic[state]


def main():
    data=pd.read_pickle('./data/train_data.pd')
    cols=data.columns
    names=['Year','Month','Day','Max_Temp','Min_Temp','Precip','Avg_Temp','Precip_cm','Humidity','MoLS']
    files=glob('./Capital Cities/*')
    li=[]
    for fil in files:
        data=pd.read_csv(fil, index_col=None, header=0, names=names)
        maca=fil.split('rcp')[-1].split('_')[0]
        loc=fil.split('_')[-1].split('.')[0]
        val=maca+','+state_dic(loc)
        data['Location'] = pd.Series([ val for x in range(len(data.index))])
        data=data[cols]
        li.append(data)
    capital_cities=pd.concat(li, axis=0, ignore_index=True)
    capital_cities.Precip=capital_cities.Precip/10
    capital_cities=capital_cities[~capital_cities.Location.str.contains('Arizona')]
    capital_cities=capital_cities[~capital_cities.Location.str.contains('California')]
    capital_cities=capital_cities[~capital_cities.Location.str.contains('Texas')]
    capital_cities=capital_cities[~capital_cities.Location.str.contains('Wisconsin')]
    capital_cities=capital_cities[~capital_cities.Location.str.contains('Minnesota')]
    capital_cities=capital_cities[~capital_cities.Location.str.contains('Delaware')]
    capital_cities=capital_cities[~capital_cities.Location.str.contains('New Jersey')]
    capital_cities=capital_cities[~capital_cities.Location.str.contains('North Carolina')]
    capital_cities.to_pickle('./data/capitals.pd')

if __name__ == '__main__':
    main()
