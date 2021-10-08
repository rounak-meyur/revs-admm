# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:50:32 2021

@author: Rounak Meyur and Swapna Thorve
"""

import sys,os
import numpy as np

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
from pyNetworklib import powerflow
print("Imported modules")


#%% Main code

homepath = inppath + "VA121_20140723.csv"
solarpath = inppath + "va121_20140723_ghi.csv"

sub = 121144
homes = get_home_data(homepath,solarpath)
dist = GetDistNet(distpath,sub)
for n in dist:
    dist.nodes[n]['load'] = [0.0]*24
    
homelist = [n for n in dist if n in homes]

# Cost profile taken for the off peak plan of AEP
cost = [0.096223]*5 + [0.099686] + [0.170936]*3 + [0.099686]*8 + [0.170936]*3 + [0.099686]*4
# Feed in tarriff rate for small residence owned rooftop solar installations
feed = 0.38
incentive = {h:[0.0]*24 for h in homelist}

#%% Optimize power usage schedule

def compute_power_schedule(net,homelist,cost,feed,incentive):
    
    for hid in homelist:
        # Compute load schedule
        L = Load(24,homes[hid])
        L.set_objective(cost, feed, incentive[hid])
        psch = L.solve(hid,grbpath)
        
        # Update load data in network
        net.nodes[hid]['load'] = [v for v in psch]
    return
    



#%% Iterative algorithm

N = dist.number_of_nodes() - 1
mu_lower = np.ones(shape=(N,24))
mu_upper = np.ones(shape=(N,24))
eps = 0.01
phi = 1e-4

k = 0
while(k <= 10):
    compute_power_schedule(dist,homelist,cost,feed,incentive)
    
    # run powerflow
    
    
    # update 
    k = k + 1























