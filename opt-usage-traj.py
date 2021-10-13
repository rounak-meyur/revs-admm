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

sub = 150728
homes = get_home_data(homepath,solarpath,p=0.0)
dist = GetDistNet(distpath,sub)

# sublist = [121143, 121144, 147793, 148717, 148718, 148719, 148720, 148721, 148723,
#         150353, 150589, 150638, 150692, 150722, 150723, 150724, 150725, 150726, 
#         150727, 150728]

# for sub in sublist[1:]:
#     dist = GetDistNet(distpath,sub)
#     homelist = [n for n in dist if dist.nodes[n]['label']=='H']
#     print(len([h for h in homelist if h not in homes]))



for n in dist:
    dist.nodes[n]['load'] = [0.0]*24

# Cost profile taken for the off peak plan of AEP
COST = [0.073626]*5 + [0.092313]*10 + [0.225525]*3 + [0.092313]*6
# Feed in tarriff rate for small residence owned rooftop solar installations
FEED = 0.38

#%% Optimize power usage schedule

def compute_power_schedule(net,homes,cost,feed,incentive):
    homelist = [n for n in net if net.nodes[n]['label'] == 'H']
    
    # UPDATE load model for residences
    power_schedule = {n:[0.0]*24 for n in net if net.nodes[n]['label']!='S'}
    # load = {n:[0.0]*24 for n in net if net.nodes[n]['label']!='S'}
    for hid in homelist:
        # Compute load schedule
        L = Load(24,homes[hid])
        L.set_objective(cost, feed, incentive[hid])
        psch = L.solve(hid,grbpath)
        
        # Update load data in network
        power_schedule[hid] = [v for v in psch]
        # load[hid] = [v for v in L.g]
    return power_schedule

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
    P = np.array([[iterdata['load'][n][t] for t in range(24)] \
                  for n in nodelist])
    M = np.linalg.inv(G)
    V = 1.0 - np.matmul(M,P)
    
    # Update dual 
    mu_low = (1-eps*phi)*mu_low + eps*(vmin - V)
    mu_up = (1-eps*phi)*mu_up + eps*(V - vmax)
    alpha = np.matmul(M,(mu_low-mu_up))
    
    # Update the iteration data
    iterdata["alpha"] = {n:alpha[i,:] for i,n in enumerate(nodelist)}
    iterdata["mu_low"] = {n:mu_low[i,:] for i,n in enumerate(nodelist)}
    iterdata["mu_up"] = {n:mu_up[i,:] for i,n in enumerate(nodelist)}
    iterdata["volt"] = {n:V[i,:] for i,n in enumerate(nodelist)}
    return iterdata
    



#%% Iterative algorithm

nodelist = [node for node in list(dist.nodes()) if dist.nodes[node]['label'] != 'S']
ITERDATA = {"alpha":{n:[0.0]*24 for n in nodelist},
            "mu_low":{n:[10.0]*24 for n in nodelist},
            "mu_up":{n:[9.0]*24 for n in nodelist},
            "volt": {n:[1.0]*24 for n in nodelist},
            "load":{n:[0.0]*24 for n in nodelist}}


alpha_history = {}
volt_history = {}
load_history = {}
k = 0
while(k <= 10):
    ITERDATA["load"] = compute_power_schedule(dist,homes,
                                              COST,FEED,ITERDATA["alpha"])
    # update 
    ITERDATA = powerflow(dist, ITERDATA)
    
    alpha_history[k] = ITERDATA["alpha"]
    volt_history[k] = ITERDATA["volt"]
    load_history[k] = ITERDATA["load"]
    k = k + 1

#%% Plot incentive evolution
import matplotlib.pyplot as plt

homelist = [n for n in dist if dist.nodes[n]['label'] == 'H']
H = homelist[30:33]
xarray = np.linspace(1,24,24)


fig1 = plt.figure(figsize=(20,16))
for m in range(5):
    ax1 = fig1.add_subplot(5,1,m+1)
    for i,h in enumerate(H):
        ax1.step(xarray,volt_history[m][h],label="home="+str(i+1))
        ax1.legend(ncol=3)


fig2 = plt.figure(figsize=(20,16))
for m in range(5):
    ax2 = fig2.add_subplot(5,1,m+1)
    for i,h in enumerate(H):
        ax2.step(xarray,load_history[m][h],label="home="+str(i+1))
        ax2.legend(ncol=3)


fig3 = plt.figure(figsize=(20,16))
for m in range(5):
    ax3 = fig3.add_subplot(5,1,m+1)
    for i,h in enumerate(H):
        ax3.step(xarray,alpha_history[m][h],label="home="+str(i+1))
        ax3.legend(ncol=3)
















