# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:50:32 2021

@author: Rounak Meyur and Swapna Thorve
"""

import sys,os
import numpy as np
import networkx as nx

workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = workpath + "/libs/"
inppath = workpath + "/input/"
figpath = workpath + "/figs/"
outpath = workpath + "/out/"
distpath = rootpath + "/synthetic-distribution/primnet/out/osm-primnet/"
grbpath = workpath + "/gurobi/"


sys.path.append(libpath)
from pyExtractlib import get_home_data, GetDistNet
from pyLoadlib import Load
print("Imported modules")


#%% Main code

homepath = inppath + "VA121_20140723.csv"
solarpath = inppath + "va121_20140723_ghi.csv"

sub = 121144
homes = get_home_data(homepath,solarpath)
dist = GetDistNet(distpath,sub)
for n in dist:
    dist.nodes[n]['load'] = [0.0]*24

# Cost profile taken for the off peak plan of AEP
COST = [0.096223]*5 + [0.099686] + [0.170936]*3 + [0.099686]*8 + [0.170936]*3 + [0.099686]*4
# Feed in tarriff rate for small residence owned rooftop solar installations
FEED = 0.38

#%% Optimize power usage schedule

def compute_power_schedule(net,homes,cost,feed,incentive):
    homelist = [n for n in net if net.nodes[n]['label'] == 'H' and n in homes]
    for hid in homelist:
        # Compute load schedule
        L = Load(24,homes[hid])
        L.set_objective(cost, feed, incentive[hid])
        psch = L.solve(hid,grbpath)
        
        # Update load data in network
        net.nodes[hid]['load'] = [v for v in psch]
    return

def powerflow(graph, iterdata, 
              eps = 0.01, phi = 1e-4,
              vmin = 0.95, vmax = 1.05):
    """
    Checks power flow solution and save dictionary of voltages.
    """
    # Run powerflow
    A = nx.incidence_matrix(graph,nodelist=list(graph.nodes()),
                            edgelist=list(graph.edges()),oriented=True).toarray()
    
    node_ind = [i for i,node in enumerate(graph.nodes()) \
                if graph.nodes[node]['label'] != 'S']
    nodelist = [node for node in list(graph.nodes()) \
                if graph.nodes[node]['label'] != 'S']
        
    # Load iteration data
    mu_low = np.array([iterdata["mu_low"][n] for n in nodelist])
    mu_up = np.array([iterdata["mu_up"][n] for n in nodelist])
    
    # Resistance data
    edge_r = []
    for e in graph.edges:
        try:
            edge_r.append(1.0/graph.edges[e]['r'])
        except:
            edge_r.append(1.0/1e-14)
    R = np.diag(edge_r)
    G = np.matmul(np.matmul(A,R),A.T)[node_ind,:][:,node_ind]
    
    # Power consumption and node voltage over time window
    P = np.array([[graph.nodes[n]['load'][t] for t in range(24)] \
                  for n in nodelist])
    M = np.linalg.inv(G)
    V = 1.0 - np.matmul(M,P)
    
    # Update dual 
    mu_low = (1-eps*phi)*mu_low + eps*(vmin - V)
    mu_up = (1-eps*phi)*mu_up + eps*(V - vmax)
    alpha = np.matmul(M,(mu_low-mu_up))
    
    iterdata = {"alpha": {n:alpha[i,:] for i,n in enumerate(nodelist)},
                "mu_low": {n:mu_low[i,:] for i,n in enumerate(nodelist)},
                "mu_up": {n:mu_up[i,:] for i,n in enumerate(nodelist)}}
    return iterdata
    



#%% Iterative algorithm

nodelist = [node for node in list(dist.nodes()) if dist.nodes[node]['label'] != 'S']
ITERDATA = {"alpha":{n:[0.0]*24 for n in nodelist},
            "mu_low":{n:[1.0]*24 for n in nodelist},
            "mu_up":{n:[1.0]*24 for n in nodelist}}


history = {}
k = 0
while(k <= 10):
    compute_power_schedule(dist,homes,COST,FEED,ITERDATA["alpha"])
    # update 
    ITERDATA = powerflow(dist, ITERDATA)
    
    history[k] = ITERDATA["alpha"]
    k = k + 1

#%% Plot incentive evolution
import matplotlib.pyplot as plt

homelist = [n for n in dist if n in homes]
hist_data = {h:[history[m][h] for m in range(11)] for h in homelist}

fig = plt.figure(figsize=(20,16))
ax = fig.add_subplot(111)
xarray = np.linspace(1,24,24)
for m in range(11):
    ax.step(xarray,history[m][homelist[10]],label="iter="+str(m+1))
ax.legend(ncol=6)


















