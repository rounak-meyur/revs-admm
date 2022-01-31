# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:50:32 2021

@author: Rounak Meyur and Swapna Thorve
"""

import sys,os
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches


workpath = os.getcwd()
libpath = workpath + "/libs/"
figpath = workpath + "/figs/"
outpath = workpath + "/out/"
distpath = workpath + "/input/osm-primnet/"
grbpath = workpath + "/gurobi/"
homepath = workpath + "/input/load/121-home-load.csv"


sys.path.append(libpath)
from pyDrawNetworklib import DrawNodes,DrawEdges

print("Imported modules")

def get_home_data(home_filename,shift=6):
    # Extract residence device data
    df_homes = pd.read_csv(home_filename)
    home_rawdata = df_homes.set_index('hid').T.to_dict()
    home_data = {h: {} for h in home_rawdata}
    
    for h in home_rawdata:
        net_load = [1e-3*home_rawdata[h]["hour"+str(i+1)] for i in range(24)]
        home_data[h]["LOAD"] = np.roll(net_load,-shift).tolist()
    return home_data

def GetDistNet(path,code):
    """
    Read the txt file containing the edgelist of the generated synthetic network and
    generates the corresponding networkx graph. The graph has the necessary node and
    edge attributes.
    
    Inputs:
        path: name of the directory
        code: substation ID or list of substation IDs
        
    Output:
        graph: networkx graph
        node attributes of graph:
            cord: longitude,latitude information of each node
            label: 'H' for home, 'T' for transformer, 'R' for road node, 
                    'S' for subs
            voltage: node voltage in pu
        edge attributes of graph:
            label: 'P' for primary, 'S' for secondary, 'E' for feeder lines
            r: resistance of edge
            x: reactance of edge
            geometry: shapely geometry of edge
            geo_length: length of edge in meters
            flow: power flowing in kVA through edge
    """
    if type(code) == list:
        graph = nx.Graph()
        for c in code:
            g = nx.read_gpickle(path+str(c)+'-dist-net.gpickle')
            graph = nx.compose(graph,g)
    else:
        graph = nx.read_gpickle(path+str(code)+'-dist-net.gpickle')
    return graph


s = 6
all_homes = get_home_data(homepath,shift=s)
dist = GetDistNet(distpath,121144)

res = [n for n in dist if dist.nodes[n]['label']=='H']

# Area of interest where EV adoption is studied
xmin = -80.37217
ymin = 37.1993
xmax = xmin + 0.010
ymax = ymin + 0.005
res_interest = [n for n in res if (xmin<=dist.nodes[n]['cord'][0]<=xmax) and (ymin<=dist.nodes[n]['cord'][1]<=ymax)]


#%% Parameters
np.random.seed(1234)
adoption = 0.1
num_choice = int(adoption * len(res_interest))
ev_home = np.random.choice(res_interest,num_choice,replace=False)

x_int = [dist.nodes[n]['cord'][0] for n in ev_home]
y_int = [dist.nodes[n]['cord'][1] for n in ev_home]

for n in dist:
    if n in ev_home:
        dist.nodes[n]["has_EV"] = True
    else:
        dist.nodes[n]["has_EV"] = False

homes = {}
for h in res:
    homes[h] = all_homes[h]
    if h in ev_home:
        homes[h]["EV"] = {"rating":2.0,"capacity":16.0,"initial":0.5,
                                         "final":0.9,"start":11,"end":23}
    else:
        homes[h]["EV"] = {}

# Other input parameters
COST = [1] * 24
prefix = "agentEV"+"home"+str(int(100*adoption))



#%% Distributed algorithm
from pySchedEVChargelib import Home,Utility


P_est = {0:{h:homes[h]["LOAD"] for h in res}}
P_sch = {0:{h:homes[h]["LOAD"] for h in res}}
G = {0:{h:[0]*len(COST) for h in res}}
S = {}
C = {}

iter_max = 20
kappa = 5.0
diff = {}


k = 0
eps = 1
while(k <= iter_max):
    # solve utility level problem to get estimate
    U_obj = Utility(dist,P_est[k],P_sch[k],G[k],kappa=kappa,low=0.9)
    U_obj.solve(grbpath)
    P_est[k+1] = U_obj.g_opt
    
    
    # solve individual agent level problem
    P_sch[k+1] = {}
    S[k+1] = {}
    C[k+1] = {}
    for h in res:
        H_obj = Home(COST,homes[h],P_est[k][h],P_sch[k][h],G[k][h],kappa=kappa)
        H_obj.solve(grbpath)
        P_sch[k+1][h] = H_obj.g_opt
        S[k+1][h] = H_obj.p_opt
        C[k+1][h] = H_obj.s_opt
        
    
    
    # update dual variables
    G[k+1] = {}
    diff[k+1] = {}
    for h in res:
        check = [(P_est[k+1][h][t] - P_sch[k+1][h][t]) for t in range(len(COST))]
        G[k+1][h] = [G[k][h][t] + (kappa/2) * check[t] for t in range(len(COST))]
        diff[k+1][h] = np.linalg.norm(np.array(check))/len(COST)
    
    
    
    k = k + 1
    print("Iteration: ",k)



#%% Plot the convergence
xtix = range(1,k+1)
fig4 = plt.figure(figsize=(20,16))
ax4 = fig4.add_subplot(1,1,1)
for h in ev_home:
    ax4.plot(xtix,[diff[k][h] for k in xtix],
             label="Home "+str(h))
ax4.set_ylabel("Difference",fontsize=25)
ax4.set_xlabel("Iterations",fontsize=25)
ax4.legend(loc='best',ncol=1,prop={'size': 25})
ax4.set_xticks(list(range(0,21,5)))
ax4.tick_params(axis='y', labelsize=25)
ax4.tick_params(axis='x', labelsize=25)
fig4.savefig("{}{}.png".format(figpath,prefix+'-convergence'),
              bbox_inches='tight')


#%% Voltage impact
from pySchedLoadlib import compute_Rmat
R = compute_Rmat(dist)
nodelist = [n for n in dist if dist.nodes[n]['label']!='S']


P = np.zeros(shape=(len(nodelist),len(COST)))
Z = np.ones(shape=(len(nodelist),len(COST)))
for i,n in enumerate(nodelist):
    if n in res:
        P[i,:] = np.array(P_sch[k][n])

V = np.sqrt(Z - R@P)
volt = {h:V[i,:].tolist() for i,h in enumerate(nodelist) if h in nodelist}
volt_data = [[volt[n][t] for n in res] for t in range(len(COST))]

fig5 = plt.figure(figsize=(20,16))
ax5 = fig5.add_subplot(1,1,1)
ax5.boxplot(volt_data)









