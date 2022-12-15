# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 12:37:23 2022

@author: Rounak Meyur
"""

import sys,os
import numpy as np


workpath = os.getcwd()
libpath = workpath + "/libs/"
grbpath = workpath + "/gurobi/"
outpath = workpath + "/out/"
compath = workpath + "/input/"
homepath = workpath + "/input/121-home-load.csv"
distpath = workpath + "/input/"
figpath = workpath + "/figs/"


sys.path.append(libpath)
from pyExtractlib import get_home_load,GetDistNet
from pySchedEVChargelib import Central

print("Imported modules and libraries")

#%% Arguments and inputs
comp_ind = 2
a = 60
r = 4800
seed = 1234
capacity = 20
initial = 0.2
start = 11
end = 23

#%% Other input parameters
s = 6
sub = 121144
all_homes = get_home_load(homepath,shift=s)
dist = GetDistNet(distpath,sub)
res = [n for n in dist if dist.nodes[n]['label']=='H']

with open(compath+str(sub)+"-com.txt",'r') as f:
    lines = f.readlines()
res_interest = [int(x) for x in lines[int(comp_ind)-1].strip('\n').split(' ')]

np.random.seed(int(seed))
adoption = float(a)/100.0
rating = float(r)*1e-3
num_choice = int(adoption * len(res_interest))
ev_home = np.random.choice(res_interest,num_choice,replace=False)

homes = {}
for h in res:
    homes[h] = {}
    homes[h]["LOAD"] = [l for l in all_homes[h]["LOAD"]]
    if h in ev_home:
        homes[h]["EV"] = {"rating":rating,"capacity":float(capacity),
                          "initial":initial,"start":start,"end":end}
    else:
        homes[h]["EV"] = {}

# Other input parameters from DOMINION Energy
COST = [0.078660]*5 + [0.095111]*10 + [0.214357]*3 + [0.095111]*6
COST = np.roll(COST,-s).tolist()
vset = 1.03

#%% Functions
def get_power_data(path):
    # Get the power consumption data from the txt files
    with open(path,'r') as f:
        lines = f.readlines()
    sepind = [i+1 for i,l in enumerate(lines) if l.strip("\n").endswith("##")]
    res_lines = lines[sepind[1]:sepind[2]-1]
    p = get_data(res_lines)
    return p

def get_soc_data(path):
    with open(path,'r') as f:
        lines = f.readlines()
    sepind = [i+1 for i,l in enumerate(lines) if l.strip("\n").endswith("##")]
    soc_lines = lines[sepind[5]:sepind[6]-1]
    p = get_data(soc_lines)
    return p


def get_data(datalines):
    dict_data = {}
    for temp in datalines:
        h = int(temp.split('\t')[0][:-1])
        dict_data[h] = [float(x) \
                        for x in temp.split('\t')[1].strip('\n').split(' ')]
    return dict_data

#%% Run simulation with centralized scheduling
C_obj = Central(homes,dist,COST,vset=vset,vmin=0.9)
C_obj.solve(grbpath)
P_res = C_obj.g_opt
SOC = C_obj.s_opt
P_ev = C_obj.p_opt



#%% Get distributed optimization output from stored result
prefix = "distEV-"+str(a)+"-adopt"+str(r)+"Watts-seed-"+str(seed)+".txt"
dirname = str(sub)+"-com-"+str(comp_ind)+"/"
P_opt = get_power_data(outpath+dirname+prefix)


#%% Deviation from the true optimal

C1 = np.array([sum([P_res[n][t]*COST[t] for t in range(24)]) for n in ev_home])
C2 = np.array([sum([P_opt[n][t]*COST[t] for t in range(24)]) for n in ev_home])
dev = 100*(C2-C1)/C1 

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(20,8))
ax = fig.add_subplot(1,1,1)
ax.bar(range(1,len(dev)+1),dev,color='r')
ax.tick_params(axis='y',labelsize=20)
ax.tick_params(bottom=False,labelbottom=False)
ax.set_ylabel("Percentage Deviation in Optimal Cost (in %)",fontsize=20)
ax.set_xlabel("Residences",fontsize=20)
fig.savefig(figpath+"cent-dist-dev.png",bbox_inches="tight")


