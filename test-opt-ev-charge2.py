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
from pyExtractlib import GetDistNet
from pySchedEVChargelib import compute_Rmat
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




#%% Extract results

sub = 121144
s = 6
T = 24
vset=1.0



#%% Get the data

# all_homes = get_home_load(homepath,shift=s)
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

R = compute_Rmat(dist)
nodelist = [n for n in dist if dist.nodes[n]['label']!='S']



#%% Multiple voltage profiles
ratings = [800, 1000, 1200, 1600, 2000]



for a in [10,20,30,40,50]:
    volt_data = {'voltage':[],'hour':[],'rating':[]}
    for r in ratings:
        # Extract data
        prefix = "agentEV-"+str(a)+"-adopt"+str(int(r))+"Watts.txt"
        with open(outpath+prefix,'r') as f:
            lines = f.readlines()
        
        sepind = [i+1 for i,l in enumerate(lines) if l.strip("\n").endswith("##")]
        res_lines = lines[sepind[1]:sepind[2]-1]
        P_res = get_data(res_lines)
        
        # Compute voltages at nodes
        P = np.zeros(shape=(len(nodelist),T))
        Z = np.ones(shape=(len(nodelist),T)) * vset
        for i,n in enumerate(nodelist):
            if n in res:
                P[i,:] = np.array(P_res[n])
        
        V = np.sqrt(Z - R@P)
        volt = {h:V[i,:].tolist() for i,h in enumerate(nodelist) if h in nodelist}
        for t in range(11,24):
            for n in res_interest:
                volt_data['voltage'].append(volt[n][t])
                volt_data['hour'].append(str((t+s)%24)+":00")
                volt_data['rating'].append(str(r)+" Watts")
        
    # Construct the pandas dataframe
    df_volt = pd.DataFrame(volt_data)
    
    fig = plt.figure(figsize=(60,20))
    ax = fig.add_subplot(1,1,1)
    ax = sns.boxplot(x="hour", y="voltage", hue="rating",
                 data=df_volt, palette="Set3",ax=ax)
    ax.set_title("EV adoption percentage: "+str(a)+"%",fontsize=50)
    ax.tick_params(axis='y',labelsize=50)
    ax.tick_params(axis='x',labelsize=50)
    ax.set_ylabel("Node Voltage",fontsize=50)
    ax.set_xlabel("Hours",fontsize=50)
    ax.legend(prop={'size': 40})
    
    
    
    fig.savefig("{}{}.png".format(figpath,str(sub)+"-adoption-"+str(a)+"-boxplot"),
                bbox_inches='tight')
        
        
    
    









