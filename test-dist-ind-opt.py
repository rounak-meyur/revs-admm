# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 10:50:53 2022

Author: Rounak Meyur

Description: Compares loading level and voltage at different nodes in the 
network for different levels of EV adoption
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import networkx as nx
from matplotlib.lines import Line2D

workpath = os.getcwd()
libpath = workpath + "/libs/"
figpath = workpath + "/figs/"
outpath = workpath + "/out/"
distpath = workpath + "/input/"
grbpath = workpath + "/gurobi/"
homepath = workpath + "/input/load/121-home-load.csv"


sys.path.append(libpath)
from pyExtractlib import GetDistNet
from pySchedEVChargelib import compute_Rmat
from pyDrawNetworklib import DrawNodes,DrawEdges
print("Imported modules and libraries")

#%% Functions for plots
def plot_network(ax,net,ev_home=[]):
    """
    """
    c_hv = "dodgerblue"
    c_prim = "green"
    c_sec = "crimson"
    c_ev = "orange"
    # Draw edges
    DrawEdges(net,ax,label='P',color='black',width=2.0)
    DrawEdges(net,ax,label='E',color=c_hv,width=2.0)
    DrawEdges(net,ax,label='S',color=c_sec,width=2.0)
    
    # Draw nodes
    DrawNodes(net,ax,label='S',color=c_hv,size=2000)
    DrawNodes(net,ax,label='T',color=c_prim,size=300)
    DrawNodes(net,ax,label='R',color='black',size=300)
    DrawNodes(net,ax,label='H',color=c_sec,size=100)
    
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    
    leghands = [#Line2D([0], [0], color=c_hv,markersize=0,
                       #label='High Voltage Feeder'),
                Line2D([0], [0], color='black', markersize=0,
                       label='Primary Network'),
                Line2D([0], [0], color=c_sec,markersize=0,
                       label='Secondary Network'),
                Line2D([0], [0], color='white', markerfacecolor=c_hv, 
                       marker='o',markersize=30,label='Substation'),
                Line2D([0], [0], color='white', markerfacecolor=c_prim, 
                       marker='o',markersize=30,label='Transformer'),
                Line2D([0], [0], color='white', markerfacecolor=c_sec, 
                       marker='o',markersize=30,label='Residence')]
    
    if len(ev_home) != 0:
        # Add the nodes where EV adoption
        x_int = [net.nodes[n]['cord'][0] for n in ev_home]
        y_int = [net.nodes[n]['cord'][1] for n in ev_home]
        ax.scatter(x_int,y_int,s=400.0,c=c_ev)
        leghands.append(Line2D([0], [0], color='white', markerfacecolor=c_ev, 
               marker='o',markersize=20,label='Residence with EV'))
    ax.legend(handles=leghands,ncol=1,prop={'size': 55},loc="lower right")
    return ax


def draw_boxplot(df,ax=None,a=None,r=None,val="voltage",colset="Set2"):
    if ax == None:
        fig = plt.figure(figsize=(60,20))
        ax = fig.add_subplot(1,1,1)
    ax = sns.boxplot(x="hour", y=val, hue="group",
                 data=df, palette=colset,ax=ax)
    # if r==None and a==None:
    #     ax.set_title("")
    # if r==None and a!=None:
    #     ax.set_title("EV adoption percentage: "+str(a)+"%",
    #                  fontsize=60)
    # if r!=None and a==None:
    #     ax.set_title("EV charger rating: "+str(r)+" Watts",
    #                  fontsize=60)
    # if r!=None and a!=None:
    #     ax.set_title("EV adoption percentage: "+str(a)+"%",
    #                  fontsize=60)
    
    ax.tick_params(axis='y',labelsize=70)
    ax.tick_params(axis='x',rotation=45,labelsize=70)
    if val == 'voltage':
        ax.set_ylabel("Node voltage (p.u.)",fontsize=80)
        # ax.set_title("Residence node voltages",fontsize=60)
    elif val == 'loading':
        ax.set_ylabel("Line loading level (%)",fontsize=80)
        # ax.set_title("Distribution line loading level",fontsize=60)
    ax.set_xlabel("Hours",fontsize=80)
    ax.legend(ncol=3,prop={'size': 80})
    return ax

def plot_convergence(ax,diff_iter):
    num_iter = len(list(diff_iter.values())[0])
    ev_homes = [h for h in diff_iter]
    xtix = range(1,num_iter+1)
    for h in ev_homes:
        ax.plot(xtix,[diff_iter[h][k] for k in range(num_iter)])
    ax.set_ylabel("Difference",fontsize=25)
    ax.set_xlabel("Iterations",fontsize=25)
    ax.set_xticks(list(range(0,num_iter+1,5)))
    ax.tick_params(axis='y',labelsize=25)
    ax.tick_params(axis='x',labelsize=25)
    return ax


#%% Functions for comparisons
def get_community(s):
    graph = GetDistNet(distpath,s)
    graph.remove_node(s)
    home_data = '\n'.join([' '.join([str(n) for n in comp if graph.nodes[n]['label']=='H']) \
                       for comp in nx.connected_components(graph)])
    with open(workpath+"/input/"+str(s)+"-com.txt",'w') as f:
        f.write(home_data)
    return

def get_obs_nodes(sub,res_ev):
    graph = GetDistNet(distpath,sub)
    reg_nodes = list(nx.neighbors(graph,sub))

    reg = []
    for h in res_ev:
        d = {n:nx.shortest_path_length(graph,h,n) for n in reg_nodes}
        reg.append(min(d, key=d.get))

    home_nodes = [n for n in graph if graph.nodes[n]['label']=='H']
    check = []
    for n in home_nodes:
        for g in list(set(reg)):
            if nx.shortest_path_length(graph,n,g) < nx.shortest_path_length(graph,n,sub):
                check.append(n)
    return check

def compute_flows(graph,p_sch):
    # Define max rating values
    RATING = {'OH_Voluta':np.sqrt(3)*95*0.24,
              'OH_Periwinkle':np.sqrt(3)*125*0.24,
              'OH_Conch':np.sqrt(3)*165*0.24,
              'OH_Neritina':np.sqrt(3)*220*0.24,
              'OH_Runcina':np.sqrt(3)*265*0.24,
              'OH_Zuzara':np.sqrt(3)*350*0.24,
              'OH_Swanate':np.sqrt(3)*145*12.47,
              'OH_Sparrow':np.sqrt(3)*185*12.47,
              'OH_Raven':np.sqrt(3)*240*12.47,
              'OH_Pegion':np.sqrt(3)*315*12.47,
              'OH_Penguin':np.sqrt(3)*365*12.47}
    
    nodelist = [n for n in graph if graph.nodes[n]['label']!='S']
    res_nodes = [n for n in graph if graph.nodes[n]['label']=='H']
    nodeind = [i for i,n in enumerate(graph.nodes) if n in nodelist]
    T = len(p_sch[res_nodes[0]])
    A = nx.incidence_matrix(graph,nodelist=graph.nodes,edgelist=graph.edges,
                            oriented=True).toarray()
    A_inv = np.linalg.inv(A[nodeind,:])
    P = np.zeros(shape=(len(nodelist),T))
    for i,n in enumerate(nodelist):
        if n in res_nodes:
            P[i,:] = np.array(p_sch[n])
    
    
    F = A_inv @ P
    rating = {e:RATING[graph.edges[e]['type']] for i,e in enumerate(graph.edges)}
    flows = {e:(F[i,:]/rating[e]).tolist() for i,e in enumerate(graph.edges)}
    return flows

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


def get_data(datalines):
    dict_data = {}
    for temp in datalines:
        h = int(temp.split('\t')[0][:-1])
        dict_data[h] = [float(x) \
                        for x in temp.split('\t')[1].strip('\n').split(' ')]
    return dict_data


def compare_method(path,adopt,rate,graph,node_interest,seed = [1234],
                   start=11,end=23,shift=6,ax=None):
    # Initialize data for pandas dataframe
    data = {'voltage':[],'hour':[],'group':[]}
    
    # Fill in the dictionary for plot data
    for j in seed:
        # Get voltage for ADMM
        prefix = "distEV-"+str(adopt)+"-adopt"+str(int(rate))\
            +"Watts-seed-"+str(j)+".txt"
        p_opt = get_power_data(path+prefix)
        volt_opt = compute_voltage(graph, p_opt)
        # Get voltage for individual optimization
        prefix = "indEV-"+str(adopt)+"-adopt"+str(int(rate))\
            +"Watts-seed-"+str(j)+".txt" 
        p_ind = get_power_data(path+prefix)
        volt_ind = compute_voltage(graph, p_ind)
        
        for t in range(start,end+1):
            for n in node_interest:
                hr = str((t+shift-1)%24)+":00 - "+str((t+shift)%24)+":00"
                data['voltage'].append(volt_opt[n][t])
                data['hour'].append(hr)
                data['group'].append("Distributed Optimization")
                data['voltage'].append(volt_ind[n][t])
                data['hour'].append(hr)
                data['group'].append("Individual Optimization")
    df = pd.DataFrame(data)
    ax = draw_boxplot(df,ax=ax,a=adopt,r=rate,colset="Set3")
    return ax


def compare_rating(path,adopt,rating_list,graph,node_interest,seed = [1234],
                   start=11,end=23,shift=6,ax=None,method="dist"):
    if method not in ["dist","ind"]:
        print("Invalid method identifier!!!")
        sys.exit(0)
    
    # Initialize data for pandas dataframe
    data = {'voltage':[],'hour':[],'group':[]}
    
    # Iterate through ratings
    for rate in rating_list:
        for j in seed:
            prefix = method+"EV-"+str(adopt)+"-adopt"+str(int(rate))\
                +"Watts-seed-"+str(j)+".txt" 
            p = get_power_data(path+prefix)
            v = compute_voltage(graph, p)
            
            # Fill in dictionary with data
            for t in range(start,end+1):
                hr = str((t+shift-1)%24)+":00 - "+str((t+shift)%24)+":00"
                for n in node_interest:
                    data['voltage'].append(v[n][t])
                    data['hour'].append(hr)
                    data['group'].append(str(rate)+" Watts")
    
    # Construct the dataframe from dictionary
    df = pd.DataFrame(data)
    ax = draw_boxplot(df,ax=ax,a=adopt)
    return ax

def compare_adoption(path,adopt_list,rate,graph,node_interest,seed = [1234],
                     start=11,end=23,shift=6,ax=None,method="dist"):
    if method not in ["dist","ind"]:
        print("Invalid method identifier!!!")
        sys.exit(0)
    
    # Initialize data for pandas dataframe
    data = {'voltage':[],'hour':[],'group':[]}
    
    # Iterate through adoption list
    for adopt in adopt_list:
        for j in seed:
            prefix = method+"EV-"+str(adopt)+"-adopt"+str(int(rate))\
                +"Watts-seed-"+str(j)+".txt"
            p = get_power_data(path+prefix)
            v = compute_voltage(graph, p)
            
            # Fill in dictionary with data
            for t in range(start,end+1):
                hr = str((t+shift-1)%24)+":00 - "+str((t+shift)%24)+":00"
                for n in node_interest:
                    data['voltage'].append(v[n][t])
                    data['hour'].append(hr)
                    data['group'].append(str(adopt)+"%")
    
    # Construct the dataframe from dictionary
    df = pd.DataFrame(data)
    ax = draw_boxplot(df,ax=ax,r=rate,colset = "Set3")
    return ax

def compare_method_flows(path,adopt,rate,graph,seed=[1234],
                         start=11,end=23,shift=6,ax=None):
    # Initialize data for pandas dataframe
    data = {'loading':[],'hour':[],'group':[]}
    
    # Fill in the dictionary for plot data
    for j in seed:
        # Get flows for ADMM
        prefix = "distEV-"+str(adopt)+"-adopt"+str(int(rate))\
            +"Watts-seed-"+str(j)+".txt" 
        p_dist = get_power_data(path+prefix)
        f_dist = compute_flows(graph,p_dist)
        
        
        # Get flows for individual optimization
        prefix = "indEV-"+str(adopt)+"-adopt"+str(int(rate))\
            +"Watts-seed-"+str(j)+".txt"
        p_ind = get_power_data(path+prefix)
        f_ind = compute_flows(graph,p_ind)
        for t in range(start,end+1):
            hr = str((t+shift-1)%24)+":00 - "+str((t+shift)%24)+":00"
            for e in graph.edges:
                str((t+shift-1)%24)+":00 - "+str((t+shift)%24)+":00"
                data['loading'].append(abs(f_dist[e][t])*100.0)
                data['hour'].append(hr)
                data['group'].append("Distributed Optimization")
                data['loading'].append(abs(f_ind[e][t])*100.0)
                data['hour'].append(hr)
                data['group'].append("Individual Optimization")
    df = pd.DataFrame(data)
    ax = draw_boxplot(df,ax=ax,a=adopt,r=rate,val="loading",colset = "Set2")
    return ax


#%% Main Code
# Some constant inputs
sub = 121144
shft = 6
T = 24
capacity = 20
initial = 0.2
start_time = 11
end_time = 23

# Cost profile
COST = [0.078660]*5 + [0.095111]*10 + [0.214357]*3 + [0.095111]*6
COST = np.roll(COST,-shft).tolist()


# Get input data
dist = GetDistNet(distpath,sub)
print("Loaded network and home data")

#%% Draw the network with community of residences
for com in range(1,6):
    dirname = str(sub)+"-com-"+str(com)+"/"
    with open(workpath+"/input/"+str(sub)+"-com.txt",'r') as f:
        lines = f.readlines()
    com_homes = [int(x) for x in lines[com-1].strip('\n').split(' ')]
    
    fig = plt.figure(figsize=(40,40))
    ax = fig.add_subplot(1,1,1)
    ax = plot_network(ax,dist,com_homes)
    
    fig.savefig(figpath+str(sub)+"-com-"+str(com)+"-homes.png",
                bbox_inches="tight")


sys.exit(0)
#%% Run for single test case
rate = 4800
adopt = 90

com = 2
dirname = str(sub)+"-com-"+str(com)+"/"
with open(workpath+"/input/"+str(sub)+"-com.txt",'r') as f:
    lines = f.readlines()
com_homes = [int(x) for x in lines[com-1].strip('\n').split(' ')]


##%% Method Comparison
fig1 = plt.figure(figsize=(60,20))
ax1 = fig1.add_subplot(1,1,1)
ax1 = compare_method(outpath+dirname,adopt,rate,dist,com_homes,ax=ax1)
fig1.savefig(figpath+str(sub)+"-com-"+str(com)+"-adopt-"+str(adopt)+"-rate-"+str(rate)+"-voltage.png",
            bbox_inches='tight')


fig2 = plt.figure(figsize=(60,20))
ax2 = fig2.add_subplot(1,1,1)
ax2 = compare_method_flows(outpath+dirname,adopt,rate,dist,ax=ax2)
fig2.savefig(figpath+str(sub)+"-com-"+str(com)+"-adopt-"+str(adopt)+"-rate-"+str(rate)+"-loading.png",
            bbox_inches='tight')

sys.exit(0)
#%% Run for multiple test cases
ratings = [3600, 4800, 6000]
adoptions = [30, 60, 90]

com = 5
dirname = str(sub)+"-com-"+str(com)+"/"
with open(workpath+"/input/"+str(sub)+"-com.txt",'r') as f:
    lines = f.readlines()
com_homes = [int(x) for x in lines[com-1].strip('\n').split(' ')]

fig = plt.figure(figsize=(40,40), dpi=72)
ax = fig.add_subplot(111)
ax = plot_network(ax,dist,ev_home=com_homes)
fig.savefig(figpath+str(sub)+"-com-"+str(com)+"-homes.png",
            bbox_inches='tight')


##%% Method Comparison
fig1 = plt.figure(figsize=(60,25*len(adoptions)))
r = 4800
for i,a in enumerate(adoptions):
    ax1 = fig1.add_subplot(len(adoptions),1,i+1)
    ax1 = compare_method(outpath+dirname,a,r,dist,com_homes,ax=ax1)
fig1.savefig(figpath+str(sub)+"-com-"+str(com)+"-dist-ind-compare-voltage.png",
            bbox_inches='tight')


fig2 = plt.figure(figsize=(60,20*len(adoptions)))
for i,a in enumerate(adoptions):
    ax2 = fig2.add_subplot(len(adoptions),1,i+1)
    ax2 = compare_method_flows(outpath+dirname,a,r,dist,ax=ax2)
fig2.savefig(figpath+str(sub)+"-com-"+str(com)+"-dist-ind-compare-loading.png",
            bbox_inches='tight')


sys.exit(0)

#%% Distributed method: compare between parameters
fig = plt.figure(figsize=(60,20*len(ratings)),dpi=72)
for i,r in enumerate(ratings):
    ax = fig.add_subplot(len(ratings),1,i+1)
    ax = compare_adoption(outpath+dirname,adoptions,r,dist,com_homes,
                        method='dist',ax=ax)
fig.savefig(figpath+str(sub)+"-com-"+str(com)+"-dist-compare-adoption.png",
            bbox_inches='tight')

fig = plt.figure(figsize=(60,20*len(adoptions)),dpi=72)
for i,a in enumerate(adoptions):
    ax = fig.add_subplot(len(adoptions),1,i+1)
    ax = compare_rating(outpath+dirname,a,ratings,dist,com_homes,
                        method='dist',ax=ax)
fig.savefig(figpath+str(sub)+"-com-"+str(com)+"-dist-compare-rating.png",
            bbox_inches='tight')

#%% Individual method: compare between parameters
fig = plt.figure(figsize=(60,20*len(adoptions)))
for i,a in enumerate(adoptions):
    ax = fig.add_subplot(len(adoptions),1,i+1)
    ax = compare_rating(outpath+dirname,a,ratings,dist,com_homes,
                        method='ind',ax=ax)
fig.savefig(figpath+str(sub)+"-com-"+str(com)+"-ind-compare-rating.png",
        bbox_inches='tight')

