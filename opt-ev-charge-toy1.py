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
num = 6
prefix = "agentEV"+"home"+str(num)


e_r = {(0,10):1e-12, (10,11):0.001,(11,12):0.001,(12,13):0.001,(13,14):0.001*util,
          (11,1):0.0005,(1,2):0.0005,(12,3):0.0005,(12,4):0.0005,(14,5):0.0005,(5,6):0.0005}



nx.set_edge_attributes(dist, e_r, 'r')
nx.set_edge_attributes(dist, elabel, 'label')

nx.set_node_attributes(dist, ncord, 'cord')
nx.set_node_attributes(dist, nlabel, 'label')

for e in edgelist:
    dist.edges[e]['geometry'] = LineString((ncord[e[0]],ncord[e[1]]))

# Home data
# homeid = [51121021300494]*3+[511210207001189]*3
# homeid = [511210207001189]*6
# homeid = [51121021300494]*6
homes = {k+1:all_homes[h] for k,h in enumerate(homeid)}
count = 0
for h in homes:
    count = count+1
    homes[h]["LOAD"] = np.roll(homes[h]["LOAD"],-6)
    if count <= num:
        homes[h]["EV"] = {"rating":2.0,"capacity":18.0,"initial":0.3,
                                     "final":0.9,"start":11,"end":23}
    else:
        homes[h]["EV"] = {}

#%% Distributed algorithm
from pySchedEVChargelib import Home,Utility

homelist = sorted([n for n in dist if dist.nodes[n]['label'] == 'H'])



P_est = {0:{h:[0]*len(COST) for h in homelist}}
P_sch = {0:{h:[0]*len(COST) for h in homelist}}
G = {0:{h:[0]*len(COST) for h in homelist}}
S = {}
C = {}

iter_max = 20
kappa = 5.0
diff = {}


k = 0
eps = 1
while(k <= iter_max):
    # solve utility level problem to get estimate
    U_obj = Utility(dist,P_est[k],P_sch[k],G[k],kappa=kappa,low=low,high=high)
    U_obj.solve(grbpath)
    P_est[k+1] = U_obj.g_opt
    
    
    # solve individual agent level problem
    P_sch[k+1] = {}
    S[k+1] = {}
    C[k+1] = {}
    for h in homelist:
        H_obj = Home(COST,homes[h],P_est[k][h],P_sch[k][h],G[k][h],kappa=kappa)
        H_obj.solve(grbpath)
        P_sch[k+1][h] = H_obj.g_opt
        S[k+1][h] = H_obj.p_opt
        C[k+1][h] = H_obj.s_opt
        
    
    
    # update dual variables
    G[k+1] = {}
    diff[k+1] = {}
    for h in homelist:
        check = [(P_est[k+1][h][t] - P_sch[k+1][h][t]) for t in range(len(COST))]
        G[k+1][h] = [G[k][h][t] + (kappa/2) * check[t] for t in range(len(COST))]
        diff[k+1][h] = np.linalg.norm(np.array(check))/len(COST)
    
    
    
    k = k + 1



##%% Plot incentive evolution
xarray = np.linspace(0,25,25)



fig1 = plt.figure(figsize = (20,16))
for i,h in enumerate(homelist):
    ax1 = fig1.add_subplot(6,1,i+1)
    ax1.step(xarray,[0]+S[k][h])
    ax1.set_ylabel("EV Charging (kW)",fontsize=15)
fig1.savefig("{}{}.png".format(figpath,prefix+'-toy-EV-load'),
            bbox_inches='tight')



fig2 = plt.figure(figsize=(20,16))
for i,h in enumerate(homelist):
    ax2 = fig2.add_subplot(6,1,i+1)
    ax2.step(xarray,[0]+P_sch[k][h])
    ax2.set_ylabel("Total load (kW)",fontsize=15)
fig2.savefig("{}{}.png".format(figpath,prefix+'-toy-EV-total'),
             bbox_inches='tight')


fig3 = plt.figure(figsize=(20,16))
for i,h in enumerate(homelist):
    ax3 = fig3.add_subplot(6,1,i+1)
    ax3.step(xarray,[0]+homes[h]["LOAD"].tolist())
    ax3.set_ylabel("Other Load (kW)",fontsize=15)
fig3.savefig("{}{}.png".format(figpath,prefix+'-toy-EV-other'),
             bbox_inches='tight')

xtix = range(1,k+1)
fig4 = plt.figure(figsize=(20,16))
ax4 = fig4.add_subplot(1,1,1)
for h in homelist:
    ax4.plot(xtix,[diff[k][h] for k in xtix],
             label="Agent "+str(h))
ax4.set_ylabel("Difference",fontsize=25)
ax4.set_xlabel("Iterations",fontsize=25)
ax4.legend(loc='best',ncol=1,prop={'size': 25})
ax4.set_xticks(list(range(0,21,5)))
ax4.tick_params(axis='y', labelsize=25)
ax4.tick_params(axis='x', labelsize=25)
fig4.savefig("{}{}.png".format(figpath,prefix+'-toy-EV-convergence'),
              bbox_inches='tight')



fig6 = plt.figure(figsize = (20,16))
for i,h in enumerate(homelist):
    ax6 = fig6.add_subplot(6,1,i+1)
    ax6.step(xarray,C[k][h])
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
    if n in homelist:
        P[i,:] = np.array(P_sch[k][n])

V = np.sqrt(Z - R@P)
volt = {h:V[i,:].tolist() for i,h in enumerate(nodelist) if h in nodelist}
fig5 = plt.figure(figsize=(20,16))
for i,h in enumerate(homelist):
    ax5 = fig5.add_subplot(6,1,i+1)
    ax5.step(xarray,[1]+volt[h])
    ax5.set_ylabel("Voltage in pu",fontsize=15)
fig5.savefig("{}{}.png".format(figpath,prefix+'-toy-usage-voltage'),
             bbox_inches='tight')














