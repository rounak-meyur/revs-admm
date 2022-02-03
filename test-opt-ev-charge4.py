# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:50:32 2021

@author: Rounak Meyur and Swapna Thorve
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

workpath = os.getcwd()
libpath = workpath + "/libs/"
figpath = workpath + "/figs/"
outpath = workpath + "/out/"
distpath = workpath + "/input/osm-primnet/"
grbpath = workpath + "/gurobi/"
homepath = workpath + "/input/load/121-home-load.csv"


sys.path.append(libpath)
from pyExtractlib import get_home_load,GetDistNet
from pySchedEVChargelib import compute_Rmat,Residence
from pyDrawNetworklib import DrawNodes,DrawEdges
print("Imported modules")

def get_data(datalines):
    dict_data = {}
    for temp in datalines:
        h = int(temp.split('\t')[0][:-1])
        dict_data[h] = [float(x) \
                        for x in temp.split('\t')[1].strip('\n').split(' ')]
    return dict_data

def plot_network(ax,net,path=None,with_secnet=False):
    """
    """
    # Draw nodes
    DrawNodes(net,ax,label='S',color='dodgerblue',size=2000)
    DrawNodes(net,ax,label='T',color='green',size=25)
    DrawNodes(net,ax,label='R',color='black',size=2.0)
    if with_secnet: DrawNodes(net,ax,label='H',color='crimson',size=2.0)
    # Draw edges
    DrawEdges(net,ax,label='P',color='black',width=2.0)
    DrawEdges(net,ax,label='E',color='dodgerblue',width=2.0)
    if with_secnet: DrawEdges(net,ax,label='S',color='crimson',width=1.0)
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    return ax


def compute_voltage(graph,p_sch,vset=1.0):
    nodelist = [n for n in graph if graph.nodes[n]['label']!='S']
    R = compute_Rmat(graph)
    # Initialize voltages and power consumptions at nodes
    P = np.zeros(shape=(len(nodelist),T))
    Z = np.ones(shape=(len(nodelist),T)) * vset
    for i,n in enumerate(nodelist):
        if n in res:
            P[i,:] = np.array(p_sch[n])
    # Compute voltage
    V = np.sqrt(Z - R@P)
    volt = {h:V[i,:].tolist() for i,h in enumerate(nodelist) if h in nodelist}
    return volt

#%% Extract results
a = 10
r = 800

sub = 121144
s = 6

T = 24
initial = 0.5
final = 0.9
start = 11
end = 23

COST = [0.078660]*5 + [0.095111]*10 + [0.214357]*3 + [0.095111]*6
COST = np.roll(COST,-s).tolist()


#%% Get the data

all_homes = get_home_load(homepath,shift=s)
dist = GetDistNet(distpath,sub)

res = [n for n in dist if dist.nodes[n]['label']=='H']

# Area of interest where EV adoption is studied
xmin = -80.37217
ymin = 37.1993
xmax = xmin + 0.010
ymax = ymin + 0.005
res_interest = [n for n in res \
                if (xmin<=dist.nodes[n]['cord'][0]<=xmax) \
                    and (ymin<=dist.nodes[n]['cord'][1]<=ymax)]





#%% Multiple voltage profiles
rating = 1600


for a in range(10,51,10):
    volt_data = {'voltage':[],'hour':[],'solve':[]}
    
    # Extract data
    prefix = "agentEV-"+str(a)+"-adopt"+str(int(rating))+"Watts.txt"
    with open(outpath+prefix,'r') as f:
        lines = f.readlines()
    
    # Get distributed solution
    sepind = [i+1 for i,l in enumerate(lines) if l.strip("\n").endswith("##")]
    res_lines = lines[sepind[1]:sepind[2]-1]
    p_opt = get_data(res_lines)
    volt_opt = compute_voltage(dist, p_opt)
    
    # Get EV adopter list
    ev_home = [int(x) for x in lines[5].split('\t')[1].split(' ')]
    # Get usage schedule with individual optimization
    p_ind = {}
    for h in res:
        homedata = {}
        homedata["LOAD"] = [l for l in all_homes[h]["LOAD"]]
        if h in ev_home:
            homedata["EV"] = {"rating":1e-3*rating,"capacity":16.0,
                              "initial":initial,"final":final,
                              "start":start,"end":end}
        else:
            homedata["EV"] = {}
        
        # Run simulation with no distributed scheduling
        H_obj = Residence(COST,homedata)
        H_obj.solve(grbpath)
        p_ind[h] = H_obj.g_opt
    volt_ind = compute_voltage(dist, p_ind)
    
    
    
    for t in range(11,24):
        for n in res_interest:
            hr = str((t+s)%24)+":00"
            volt_data['voltage'].append(volt_opt[n][t])
            volt_data['hour'].append(hr)
            volt_data['solve'].append("ADMM")
            volt_data['voltage'].append(volt_ind[n][t])
            volt_data['hour'].append(hr)
            volt_data['solve'].append("no_ADMM")
        
    # Construct the pandas dataframe
    df_volt = pd.DataFrame(volt_data)
    
    fig = plt.figure(figsize=(60,20))
    ax = fig.add_subplot(1,1,1)
    ax = sns.boxplot(x="hour", y="voltage", hue="solve",
                 data=df_volt, palette="Set3",ax=ax)
    ax.set_title("EV adoption percentage: "+str(a)+"%",fontsize=50)
    ax.tick_params(axis='y',labelsize=50)
    ax.tick_params(axis='x',labelsize=50)
    ax.set_ylabel("Node Voltage",fontsize=50)
    ax.set_xlabel("Hours",fontsize=50)
    ax.legend(prop={'size': 40})
    
    fig.savefig("{}{}.png".format(figpath,
                                  str(sub)+"-admm-versus-ind-adopt"+str(a)+"-rating-"+str(rating)+"-boxplot"),
                bbox_inches='tight')
        
        
    
    









