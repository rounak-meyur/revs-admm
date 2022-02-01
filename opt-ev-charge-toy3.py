# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:50:32 2021

@author: Rounak Meyur and Swapna Thorve
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt


workpath = os.getcwd()
libpath = workpath + "/libs/"
figpath = workpath + "/figs/"
outpath = workpath + "/out/"
distpath = workpath + "/input/osm-primnet/"
grbpath = workpath + "/gurobi/"
homepath = workpath + "/input/load/121-home-load.csv"


sys.path.append(libpath)
from pyExtractlib import get_home_load,GetDistNet
from pySchedEVChargelib import compute_Rmat
print("Imported modules")

def get_data(datalines):
    dict_data = {}
    for temp in datalines:
        h = int(temp.split('\t')[0][:-1])
        dict_data[h] = [float(x) \
                        for x in temp.split('\t')[1].strip('\n').split(' ')]
    return dict_data

#%% Extract results
a = 10
r = 800

prefix = "agentEV-"+str(a)+"-adopt"+str(int(r))+"Watts.txt"
with open(outpath+prefix,'r') as f:
    lines = f.readlines()

sub = int(lines[0].split('\t')[1])
vset = float(lines[1].split('\t')[1])
s = 6
ev_home = [int(x) for x in lines[5].split('\t')[1].split(' ')]
T = 24

# Get usage profile
sepind = [i+1 for i,l in enumerate(lines) if l.strip("\n").endswith("##")]

res_lines = lines[sepind[1]:sepind[2]-1]
ev_lines = lines[sepind[3]:sepind[4]-1]
soc_lines = lines[sepind[5]:sepind[6]-1]
diff_lines = lines[sepind[7]:len(lines)]

P_res = get_data(res_lines)
P_ev = get_data(ev_lines)
SOC = get_data(soc_lines)
diff = get_data(diff_lines)




#%% Get the data

# all_homes = get_home_load(homepath,shift=s)
dist = GetDistNet(distpath,sub)

res = [n for n in dist if dist.nodes[n]['label']=='H']

# Area of interest where EV adoption is studied
# xmin = -80.37217
# ymin = 37.1993
# xmax = xmin + 0.010
# ymax = ymin + 0.005
# res_interest = [n for n in res if (xmin<=dist.nodes[n]['cord'][0]<=xmax) and (ymin<=dist.nodes[n]['cord'][1]<=ymax)]




#%% Voltage impact
R = compute_Rmat(dist)
nodelist = [n for n in dist if dist.nodes[n]['label']!='S']



#%% Multiple voltage profiles
a = 10
fig4 = plt.figure(figsize=(60,48))
fig5 = plt.figure(figsize=(60,48))
for l,r in enumerate([800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600]):
    # Extract data
    prefix = "agentEV-"+str(a)+"-adopt"+str(int(r))+"Watts.txt"
    with open(outpath+prefix,'r') as f:
        lines = f.readlines()
    
    sepind = [i+1 for i,l in enumerate(lines) if l.strip("\n").endswith("##")]
    res_lines = lines[sepind[1]:sepind[2]-1]
    diff_lines = lines[sepind[7]:len(lines)]
    P_res = get_data(res_lines)
    diff = get_data(diff_lines)
    
    # Compute voltages at nodes
    P = np.zeros(shape=(len(nodelist),T))
    Z = np.ones(shape=(len(nodelist),T)) * vset
    for i,n in enumerate(nodelist):
        if n in res:
            P[i,:] = np.array(P_res[n])
    
    V = np.sqrt(Z - R@P)
    volt = {h:V[i,:].tolist() for i,h in enumerate(nodelist) if h in nodelist}
    volt_data = [[volt[n][t] for n in res] for t in range(T)]
    
    # Add the plot to figure
    ax5 = fig5.add_subplot(3,3,l+1)
    d = ax5.boxplot(volt_data)
    
    # Plot the convergence
    num_iter = 15
    xtix = range(1,num_iter+1)
    ax4 = fig4.add_subplot(3,3,l+1)
    for h in ev_home:
        ax4.plot(xtix,[diff[h][k] for k in range(num_iter)],
                  label="Home "+str(h))
    ax4.set_ylabel("Difference")
    ax4.set_xlabel("Iterations")
    ax4.legend(loc='best',ncol=1,prop={'size': 5})
    ax4.set_xticks(list(range(0,num_iter+1,5)))
    ax4.tick_params(axis='y')
    ax4.tick_params(axis='x')
    
    









