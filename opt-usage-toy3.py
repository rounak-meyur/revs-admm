# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:50:32 2021

@author: Rounak Meyur and Swapna Thorve
"""

import sys,os
import numpy as np
import networkx as nx
import pandas as pd
from math import ceil
from shapely.geometry import LineString

workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = workpath + "/libs/"
inppath = workpath + "/input/"
figpath = workpath + "/figs/"
outpath = workpath + "/out/"
distpath = rootpath + "/synthetic-distribution/primnet/out/osm-primnet/"
grbpath = workpath + "/gurobi/"
homepath = inppath + "VA121_20140723.csv"


sys.path.append(libpath)
# from pySchedLoadlib import Central
print("Imported modules")



#%% Home definition

def get_home_data(home_filename):
    # Extract residence device data
    df_homes = pd.read_csv(home_filename)
    home_rawdata = df_homes.set_index('hid').T.to_dict()
    home_data = {h: {"PV":{},"ESS":{},"SL":{},"TCL":{},"FIXED":{},"ORG":{}} \
                 for h in home_rawdata}
    
    for i,h in enumerate(home_rawdata):
        # Schedulable loads
        if home_rawdata[h]["hasCw"]==1:
            time = home_rawdata[h]["minutesCd"] + home_rawdata[h]["minutesCw"]
            rating = ((home_rawdata[h]["wattageCd"]*home_rawdata[h]["minutesCd"])\
                      + (home_rawdata[h]["wattageCw"]*home_rawdata[h]["minutesCw"])) / time
            home_data[h]["SL"]["laundry"] = {"rating":rating/1000.0,
                                             "time":ceil(time/60.0)}
        if home_rawdata[h]["hasDw"]==1:
            time = home_rawdata[h]["minutesDw"]
            rating = home_rawdata[h]["wattageDw"]
            home_data[h]["SL"]["dwasher"] = {"rating":rating/1000.0,
                                                "time":ceil(time/60.0)}
        # if home_rawdata[h]["hasCw"]==1:
        #     home_data[h]["SL"]["laundry"] = {"rating":1.5,"time":2}
        # if home_rawdata[h]["hasDw"]==1:
        #     home_data[h]["SL"]["dwasher"] = {"rating":0.9,"time":2}
        
        # Fixed loads
        home_data[h]["FIXED"]["base"] = [home_rawdata[h]["base_load_"+str(i+1)] \
                                         for i in range(24)]
        home_data[h]["FIXED"]["hvac"] = [home_rawdata[h]["hvac_kwh_"+str(i+1)] \
                                         for i in range(24)]
        home_data[h]["FIXED"]["hoth2o"] = [home_rawdata[h]["hoth2o_kwh_"+str(i+1)] \
                                         for i in range(24)]
        
        # Original Schedulable load profile
        home_data[h]["ORG"]["dwasher"] = [home_rawdata[h]["dwasher_kwh_"+str(i+1)]\
                                          for i in range(24)]
        home_data[h]["ORG"]["laundry"] = [home_rawdata[h]["laundry_kwh_"+str(i+1)]\
                                          for i in range(24)]
        home_data[h]["ORG"]["total"] = [home_data[h]["ORG"]["dwasher"][t] \
            + home_data[h]["ORG"]["laundry"][t] + home_data[h]["FIXED"]["base"][t] \
                + home_data[h]["FIXED"]["hvac"][t] + home_data[h]["FIXED"]["hoth2o"][t] \
                    for t in range(24)]
    return home_data

all_homes = get_home_data(homepath)

# Store the home data for the following houses with IDs from the toy network
homeid = [511210207001189,51121020900342,511210203001855,511210207001113,
          51121020900346,51121021300494]


COST = [1]*24



#%% Network definition
dist = nx.Graph()
edgelist = [(0,10),(10,11),(11,12),(12,13),(13,14),(11,1),(1,2),(12,3),(12,4),
            (14,5),(5,6)]
dist.add_edges_from(edgelist)
nlabel = {0:'S',1:'H',2:'H',3:'H',4:'H',5:'H',6:'H',
          10:'R',11:'T',12:'T',13:'R',14:'T'}
ncord = {0:[0,0],1:[1,-1],2:[1,-2],3:[2,-1],4:[2,1],5:[4,-1],6:[4,-2],
         10:[0.5,0],11:[1,0],12:[2,0],13:[3,0],14:[4,0]}
elabel = {(0,10):'E',(10,11):'P',(11,12):'P',(12,13):'P',(13,14):'P',
          (11,1):'S',(1,2):'S',(12,3):'S',(12,4):'S',(14,5):'S',(5,6):'S'}

util = 5
prefix = "central-control"+"net0"+"pos0"
# prefix = "alllowload"+"net0"+"pos0"

e_r = {(0,10):1e-12, (10,11):0.001,(11,12):0.001,(12,13):0.001,(13,14):0.001*util,
          (11,1):0.0005,(1,2):0.0005,(12,3):0.0005,(12,4):0.0005,(14,5):0.0005,(5,6):0.0005}


nx.set_edge_attributes(dist, e_r, 'r')
nx.set_edge_attributes(dist, elabel, 'label')

nx.set_node_attributes(dist, ncord, 'cord')
nx.set_node_attributes(dist, nlabel, 'label')

for e in edgelist:
    dist.edges[e]['geometry'] = LineString((ncord[e[0]],ncord[e[1]]))



# Home data
homeid = [51121021300494]*4+[511210207001189]*2
# homeid = [511210207001189]*6
# homeid = [51121021300494]*6
homes = {k+1:all_homes[h] for k,h in enumerate(homeid)}

#%% Central algorithm
from pySchedLoadlib import Central
c = Central(homes,dist,COST)
c.solve(grbpath)

#Plot incentive evolution
import matplotlib.pyplot as plt

homelist = sorted([n for n in dist if dist.nodes[n]['label'] == 'H'])
xarray = np.linspace(0,25,25)



for device in ['laundry','dwasher']:
    fig = plt.figure(figsize = (20,16))
    for i,h in enumerate(homelist):
        ax = fig.add_subplot(6,1,i+1)
        if device in homes[h]["SL"]:
            ax.step(xarray,[0]+c.p_sch[h][device])
        ax.set_ylabel("Sch. Load (kW)",fontsize=15)
    fig.savefig("{}{}.png".format(figpath,prefix+'-toy-usage-'+device),
                bbox_inches='tight')


fig2 = plt.figure(figsize=(20,16))
for i,h in enumerate(homelist):
    ax2 = fig2.add_subplot(6,1,i+1)
    ax2.step(xarray,[0]+c.g_opt[h])
        
    ax2.set_ylabel("Total load (kW)",fontsize=15)
fig2.savefig("{}{}.png".format(figpath,prefix+'-toy-usage-total'),
             bbox_inches='tight')


fig3 = plt.figure(figsize=(20,16))
for i,h in enumerate(homelist):
    ax3 = fig3.add_subplot(6,1,i+1)
    ax3.step(xarray,np.insert(c.a_opt[h],0,0))
    ax3.set_ylabel("Incentive signal",fontsize=15)
fig3.savefig("{}{}.png".format(figpath,prefix+'-toy-usage-incentive'),
             bbox_inches='tight')





