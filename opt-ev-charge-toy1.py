# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:50:32 2021

@author: Rounak Meyur and Swapna Thorve
"""

import sys,os
import numpy as np
import networkx as nx
import pandas as pd
from shapely.geometry import LineString
import matplotlib.pyplot as plt

workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = workpath + "/libs/"
inppath = workpath + "/input/load/"
figpath = workpath + "/figs/"
outpath = workpath + "/out/"
distpath = rootpath + "/synthetic-distribution/primnet/out/osm-primnet/"
grbpath = workpath + "/gurobi/"
homepath = inppath + "VA121_20140723.csv"


sys.path.append(libpath)
print("Imported modules")

#%% Home definition

def get_home_data(home_filename):
    # Extract residence device data
    df_homes = pd.read_csv(home_filename)
    home_rawdata = df_homes.set_index('hid').T.to_dict()
    home_data = {h: {} for h in home_rawdata}
    
    for h in home_rawdata:
        # Fixed loads
        base = [home_rawdata[h]["base_load_"+str(i+1)] for i in range(24)]
        hvac = [home_rawdata[h]["hvac_kwh_"+str(i+1)] for i in range(24)]
        hoth2o = [home_rawdata[h]["hoth2o_kwh_"+str(i+1)] for i in range(24)]
        
        # Original Schedulable load profile
        dwasher = [home_rawdata[h]["dwasher_kwh_"+str(i+1)] for i in range(24)]
        laundry = [home_rawdata[h]["laundry_kwh_"+str(i+1)] for i in range(24)]
        home_data[h]["LOAD"] = [base[t]+hvac[t]+hoth2o[t]+dwasher[t]+laundry[t]\
                                for t in range(24)]
        
        # Add EV charger ratings
        home_data[h]["EV"] = {}
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


util = 1.0
low=0.95
high=1.05
r = 1000
prefix = "agentEV-"+str(r)+"-watts"


e_r = {(0,10):1e-12, (10,11):0.001,(11,12):0.001,(12,13):0.001,(13,14):0.001*util,
          (11,1):0.0005,(1,2):0.0005,(12,3):0.0005,(12,4):0.0005,(14,5):0.0005,(5,6):0.0005}



nx.set_edge_attributes(dist, e_r, 'r')
nx.set_edge_attributes(dist, elabel, 'label')

nx.set_node_attributes(dist, ncord, 'cord')
nx.set_node_attributes(dist, nlabel, 'label')

for e in edgelist:
    dist.edges[e]['geometry'] = LineString((ncord[e[0]],ncord[e[1]]))

# Home data
res = [n for n in dist if dist.nodes[n]['label']=='H']
ev_home = [h for h in homeid]
homes = {}
for k,h in enumerate(homeid):
    homes[k+1] = {}
    homes[k+1]["LOAD"] = [l for l in all_homes[h]["LOAD"]]
    if h in ev_home:
        homes[k+1]["EV"] = {"rating":r/1000,"capacity":16.0,
                          "initial":0.5,"final":0.9,
                          "start":11,"end":23}
    else:
        homes[k+1]["EV"] = {}

#%% Distributed algorithm
from pySchedEVChargelib import solve_ADMM
diff,P_res,P_ev,SOC = solve_ADMM(homes,dist,COST,grbpath=grbpath,
                                 vlow=low,iter_max=20)


##%% Plot incentive evolution
xarray = np.linspace(0,24,25)



fig1 = plt.figure(figsize = (20,16))
for i,h in enumerate(res):
    ax1 = fig1.add_subplot(6,1,i+1)
    ax1.step(xarray,[0]+P_ev[h])
    ax1.set_ylabel("EV Charging (kW)",fontsize=15)
fig1.savefig("{}{}.png".format(figpath,prefix+'-toy-EV-load'),
            bbox_inches='tight')



fig2 = plt.figure(figsize=(20,16))
for i,h in enumerate(res):
    ax2 = fig2.add_subplot(6,1,i+1)
    ax2.step(xarray,[0]+P_res[h])
    ax2.set_ylabel("Total load (kW)",fontsize=15)
fig2.savefig("{}{}.png".format(figpath,prefix+'-toy-EV-total'),
             bbox_inches='tight')


num_iter = len(diff)
xtix = range(1,num_iter+1)
fig4 = plt.figure(figsize=(20,16))
ax4 = fig4.add_subplot(1,1,1)
for h in res:
    ax4.plot(xtix,[diff[k][h] for k in xtix],
             label="Agent "+str(h))
ax4.set_ylabel("Difference",fontsize=25)
ax4.set_xlabel("Iterations",fontsize=25)
ax4.legend(loc='best',ncol=1,prop={'size': 25})
ax4.set_xticks(list(range(0,num_iter+1,5)))
ax4.tick_params(axis='y', labelsize=25)
ax4.tick_params(axis='x', labelsize=25)
fig4.savefig("{}{}.png".format(figpath,prefix+'-toy-EV-convergence'),
              bbox_inches='tight')



fig6 = plt.figure(figsize = (20,16))
for i,h in enumerate(res):
    ax6 = fig6.add_subplot(6,1,i+1)
    ax6.step(xarray,SOC[h])
    ax6.set_ylabel("EV Charge",fontsize=15)
fig6.savefig("{}{}.png".format(figpath,prefix+'-toy-EV-charge'),
            bbox_inches='tight')



# Check voltages
from pySchedLoadlib import compute_Rmat
R = compute_Rmat(dist)
nodelist = [n for n in dist if dist.nodes[n]['label']!='S']


P = np.zeros(shape=(len(nodelist),len(COST)))
Z = np.ones(shape=(len(nodelist),len(COST)))
for i,n in enumerate(nodelist):
    if n in res:
        P[i,:] = np.array(P_res[n])

V = np.sqrt(Z - R@P)
volt = {h:V[i,:].tolist() for i,h in enumerate(nodelist) if h in nodelist}
fig5 = plt.figure(figsize=(20,16))
for i,h in enumerate(res):
    ax5 = fig5.add_subplot(6,1,i+1)
    ax5.step(xarray,[1]+volt[h])
    ax5.set_ylabel("Voltage in pu",fontsize=15)
fig5.savefig("{}{}.png".format(figpath,prefix+'-toy-usage-voltage'),
             bbox_inches='tight')














