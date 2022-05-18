# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 10:51:15 2022

Author: Rounak Meyur

Description: Plot metric of reliability by showing number of residences with 
undervoltage or lower than acceptable voltage
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import itertools
from matplotlib.patches import Patch

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
print("Imported modules and libraries")


#%% Functions
def get_data(datalines):
    dict_data = {}
    for temp in datalines:
        h = int(temp.split('\t')[0][:-1])
        dict_data[h] = [float(x) \
                        for x in temp.split('\t')[1].strip('\n').split(' ')]
    return dict_data

def compute_voltage(graph,p_sch,vset=1.0):
    nodelist = [n for n in graph if graph.nodes[n]['label']!='S']
    res_nodes = [n for n in graph if graph.nodes[n]['label']=='H']
    T = len(p_sch[res_nodes[0]])
    R = compute_Rmat(graph)
    
    # Initialize voltages and power consumptions at nodes
    P = np.zeros(shape=(len(nodelist),T))
    Z = np.ones(shape=(len(nodelist),T)) * (vset*vset)
    
    for i,n in enumerate(nodelist):
        if n in res_nodes:
            P[i,:] = np.array(p_sch[n])
    # Compute voltage
    V = np.sqrt(Z - R@P)
    volt = {h:V[i,:].tolist() for i,h in enumerate(nodelist) if h in nodelist}
    return volt

def get_power_data(path):
    # Get the power consumption data from the txt files
    with open(path,'r') as f:
        lines = f.readlines()
    sepind = [i+1 for i,l in enumerate(lines) if l.strip("\n").endswith("##")]
    res_lines = lines[sepind[1]:sepind[2]-1]
    p = get_data(res_lines)
    return p


#%% Functions to plot the grouped bar plots

def compare_node_counts(path,adopt,rate,graph,node_interest,
                        v_range=[0.92,0.95,0.97],seed = [1234],
                        start=11,end=23,shift=6,ax=None):
    # Initialize data for pandas dataframe
    data = {'count':[],'stack':[],'hour':[],'group':[]}

    # Fill in the dictionary for plot data
    v_str = ["< "+str(v_range[0])+" p.u."] \
        + [str(v_range[i])+"-"+str(v_range[i+1])+" p.u." \
                  for i in range(len(v_range)-1)]
    
    for j in seed:
        # Get voltage for ADMM
        prefix = "distEV-"+str(adopt)+"-adopt"+str(int(rate))\
            +"Watts-seed-"+str(j)+".txt" 
        p_dist = get_power_data(path+prefix)
        volt_dist = compute_voltage(graph, p_dist)
        
        # Get voltage for individual optimization
        prefix = "indEV-"+str(adopt)+"-adopt"+str(int(rate))\
            +"Watts-seed-"+str(j)+".txt"
        p_ind = get_power_data(path+prefix)
        volt_ind = compute_voltage(graph, p_ind)


        for t in range(start,end+1):
            hr = str((t+shift-1)%24)+":00 - "+str((t+shift)%24)+":00"
            for i in range(len(v_range)):
                num_dist = len([n for n in node_interest \
                                if volt_dist[n][t]<=v_range[i]])
                data['count'].append(num_dist)
                data['stack'].append(v_str[i])
                data['hour'].append(hr)
                data['group'].append("Distributed Optimization")
                
                
                num_ind = len([n for n in node_interest \
                                if volt_ind[n][t]<=v_range[i]])
                data['count'].append(num_ind)
                data['stack'].append(v_str[i])
                data['hour'].append(hr)
                data['group'].append("Individual Optimization")
    df = pd.DataFrame(data)
    ax = draw_barplot(df,v_str,ax,adopt=adopt,rate=rate)
    return ax

def draw_barplot(df,groups,ax=None,adopt=90,rate=4800):
    if ax == None:
        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(1,1,1)
    
    # Draw the bar plot
    num_stack = len(groups)
    colors = sns.color_palette("Set3")[:num_stack]
    hours = df.hour.unique()
    for i,g in enumerate(df.groupby("stack",sort=False)):
        ax = sns.barplot(data=g[1], x="hour",y="count",hue="group",
                              palette=[colors[i]],ax=ax,
                              zorder=-i, edgecolor="k",errwidth=5)
    
    
    # Format other stuff
    ax.tick_params(axis='y',labelsize=40)
    ax.tick_params(axis='x',labelsize=40,rotation=90)
    ax.set_ylabel("Number of residences",fontsize=50)
    ax.set_xlabel("Hours",fontsize=50)
    ax.set_title("Adoption percentage: "+str(adopt)+"%",fontsize=50)
    ax.set_ylim(bottom=0,top=80)


    hatches = itertools.cycle(['/', ''])
    for i, bar in enumerate(ax.patches):
        if i%(len(hours)) == 0:
            hatch = next(hatches)
        bar.set_hatch(hatch)


    han1 = [Patch(facecolor=color, edgecolor='black', label=label) \
                  for label, color in zip(groups, colors)]
    han2 = [Patch(facecolor="white",edgecolor='black',
                  label="Distributed optimization",hatch='/'),
                   Patch(facecolor="white",edgecolor='black',
                         label="Individual optimization",hatch='')]
    leg1 = ax.legend(handles=han1,ncol=1,prop={'size': 30},loc='upper right')
    ax.legend(handles=han2,ncol=1,prop={'size': 30},loc='upper left')
    ax.add_artist(leg1)
    return ax
    

#%% Get out of limit count for single adoption
rating = 4800
sub = 121144
com = 4
dirname = str(sub)+"-com-"+str(com)+"/"
dist = GetDistNet(distpath,sub)
start = 20
end = 22
shift = 6

dirname = str(sub)+"-com-"+str(com)+"/"
with open(workpath+"/input/"+str(sub)+"-com.txt",'r') as f:
    lines = f.readlines()
res_interest = [int(x) for x in lines[com-1].strip('\n').split(' ')]



seeds = [1234,56,567,67,678,5678]
adopt_list = [30,60,90]
fig = plt.figure(figsize=(20*len(adopt_list),20))

for i,adopt in enumerate(adopt_list):
    ax = fig.add_subplot(1,len(adopt_list),i+1)
    ax = compare_node_counts(outpath+dirname,adopt,rating,dist,
                             res_interest,ax=ax,start=20,end=22,
                             v_range=[0.92,0.95,0.98],seed=seeds)

    

fig.savefig(figpath+str(sub)+"-com-"+str(com)+"-rate-"+str(rating)+"-voltlimit.png",
            bbox_inches='tight')
    

