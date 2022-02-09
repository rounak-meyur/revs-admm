# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 10:51:15 2022

@author: rm5nz
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import networkx as nx
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
                              zorder=-i, edgecolor="k")
    
    
    # Format other stuff
    ax.tick_params(axis='y',labelsize=40)
    ax.tick_params(axis='x',labelsize=40,rotation=90)
    ax.set_ylabel("Number of residences",fontsize=40)
    ax.set_xlabel("Hours",fontsize=40)
    ax.set_title("Adoption percentage: "+str(adopt)+"%",fontsize=40)
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
com = 5
dirname = str(sub)+"-com-"+str(com)+"/"
dist = GetDistNet(distpath,sub)
start = 20
end = 22
shift = 6

dirname = str(sub)+"-com-"+str(com)+"/"
with open(workpath+"/input/"+str(sub)+"-com.txt",'r') as f:
    lines = f.readlines()
res_interest = [int(x) for x in lines[com-1].strip('\n').split(' ')]



seeds = [1234,12,123,234]
# seeds = [1234]
adopt_list = [60,80,90]
fig = plt.figure(figsize=(20*len(adopt_list),20))

for i,adopt in enumerate(adopt_list):
    ax = fig.add_subplot(1,len(adopt_list),i+1)
    ax = compare_node_counts(outpath+dirname,adopt,rating,dist,
                             res_interest,ax=ax,start=20,end=22,
                             v_range=[0.92,0.95,0.98],seed=seeds)

    

fig.savefig(figpath+str(sub)+"-com-"+str(com)+"-rate-"+str(rating)+"-outlimit.png",
            bbox_inches='tight')
    


sys.exit(0)

#%% Get out of limit count for multiple adoption
rate = 4800
sub = 121144
com = 2
dirname = str(sub)+"-com-"+str(com)+"/"
path = outpath + dirname
graph = GetDistNet(distpath,sub)
start = 20
end = 22
shift = 6

dirname = str(sub)+"-com-"+str(com)+"/"
with open(workpath+"/input/"+str(sub)+"-com.txt",'r') as f:
    lines = f.readlines()
node_interest = [int(x) for x in lines[com-1].strip('\n').split(' ')]
total_res = len(node_interest)

# Initialize data for pandas dataframe
data_dist = {'count':[],'hour':[],'adopt':[]}
data_ind = {'count':[],'hour':[],'adopt':[]}

# Fill in the dictionary for plot data
adopt_list = [40,60,80,100]
for adopt in adopt_list:
    for t in range(start,end+1):
        hr = str((t+shift-1)%24)+":00 - "+str((t+shift)%24)+":00"
        
        # Get voltage for ADMM
        prefix = "distEV-"+str(adopt)+"-adopt"+str(int(rate))+"Watts.txt" 
        p_dist = get_power_data(path+prefix)
        volt_dist = compute_voltage(graph, p_dist)
        num_dist = len([n for n in node_interest if volt_dist[n][t]<=0.95])
        data_dist['count'].append(-num_dist)
        data_dist['hour'].append(hr)
        data_dist['adopt'].append(adopt)
        
        # Get voltage for individual optimization
        prefix = "indEV-"+str(adopt)+"-adopt"+str(int(rate))+"Watts.txt" 
        p_ind = get_power_data(path+prefix)
        volt_ind = compute_voltage(graph, p_ind)
        num_ind = len([n for n in node_interest if volt_ind[n][t]<=0.95])
        data_ind['count'].append(num_ind)
        data_ind['hour'].append(hr)
        data_ind['adopt'].append(adopt)


##%% Plot the bars
colors = sns.color_palette("Set3")[:len(adopt_list)]
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(1,1,1)

df_dist = pd.DataFrame(data_dist)
ax = sns.barplot(data=df_dist, x="hour",y="count",hue="adopt",
                     palette="Set3",ax=ax,edgecolor="k")

df_ind = pd.DataFrame(data_ind)
ax = sns.barplot(data=df_ind, x="hour",y="count",hue="adopt",
                     palette="Set3",ax=ax,edgecolor="k")
    
ax.tick_params(axis='y',labelsize=40)
ax.tick_params(axis='x',labelsize=40)
ax.set_ylabel("Number of residences",fontsize=40)
ax.set_xlabel("Hours",fontsize=40)
ax.axhline(color='k',linewidth=2.0)

lim = 40
ax.set_ylim(bottom=-lim,top=lim)
ypos = np.arange(-lim,lim+1,20)
ypos_label = [str(int(abs(y))) for y in ypos]
ax.set_yticks(ypos)
ax.set_yticklabels(ypos_label)


a_str = [str(x)+"%" for x in adopt_list]
leghandles = [Patch(facecolor=color, label=label) \
              for label, color in zip(a_str, colors)]
ax.legend(handles=leghandles,ncol=2,prop={'size': 40})


sys.exit(0)

