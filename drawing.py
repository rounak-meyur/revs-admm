# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:45:38 2021

Author: Rounak
Description: Functions to create network representations and color graphs based
on their attributes.
"""

from shapely.geometry import Point
import geopandas as gpd
import seaborn as sns
import numpy as np
import networkx as nx
import pandas as pd

#%% Power flow functions
def compute_Rmat(graph):
    A = nx.incidence_matrix(graph,nodelist=list(graph.nodes()),
                            edgelist=list(graph.edges()),oriented=True).toarray()
    node_ind = [i for i,node in enumerate(graph.nodes()) \
                if graph.nodes[node]['label'] != 'S']
    
    # Resistance data
    F = np.linalg.inv(A[node_ind,:].T)
    D = np.diag([graph.edges[e]['r'] for e in graph.edges])
    return 2*F@D@(F.T)

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

#%% Network Geometries
def DrawNodes(synth_graph,ax,label=['S','T','H'],color='green',size=25,
              alpha=1.0):
    """
    Get the node geometries in the network graph for the specified node label.
    """
    # Get the nodes for the specified label
    if label == []:
        nodelist = list(synth_graph.nodes())
    else:
        nodelist = [n for n in synth_graph.nodes() \
                    if synth_graph.nodes[n]['label']==label \
                        or synth_graph.nodes[n]['label'] in label]
    # Get the dataframe for node and edge geometries
    d = {'nodes':nodelist,
         'geometry':[Point(synth_graph.nodes[n]['cord']) for n in nodelist]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax,color=color,markersize=size,alpha=alpha)
    return

def DrawEdges(synth_graph,ax,label=['P','E','S'],color='black',width=2.0,
              style='solid',alpha=1.0):
    """
    """
    # Get the nodes for the specified label
    if label == []:
        edgelist = list(synth_graph.edges())
    else:
        edgelist = [e for e in synth_graph.edges() \
                    if synth_graph[e[0]][e[1]]['label']==label\
                        or synth_graph[e[0]][e[1]]['label'] in label]
    d = {'edges':edgelist,
         'geometry':[synth_graph.edges[e]['geometry'] for e in edgelist]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax=ax,edgecolor=color,linewidth=width,linestyle=style,alpha=alpha)
    return









#%% Box plots of loading level and voltages
def boxplot_flow(p_demand, dist, ax, **kwargs):
    # keyword arguments
    start = kwargs.get("start", 11)
    end = kwargs.get("end", 23)
    shift = kwargs.get("shift", 6)
    ticklabelsize = kwargs.get("tick_labelsize", 30)
    labelsize = kwargs.get("labelsize", 30)
    
    # Initialize data for pandas dataframe
    data = {'loading':[],'hour':[]}
    
    # Get flows for individual optimization
    f_ind = compute_flows(dist, p_demand)
    for t in range(start,end+1):
        hr = f"{(t+shift-1) % 24}:00 - {(t+shift) % 24}:00"
        for e in dist.edges:
            data['loading'].append(abs(f_ind[e][t])*100.0)
            data['hour'].append(hr)
    
    # construct pandas dataframe for plot
    df = pd.DataFrame(data)
    ax = sns.boxplot(x="hour", y="loading",
                 data=df, color=sns.color_palette("Set2")[0], ax=ax)
    
    ax.tick_params(axis='y',labelsize=ticklabelsize)
    ax.tick_params(axis='x',rotation=45,labelsize=ticklabelsize)
    ax.set_ylabel("Line loading level (%)",fontsize=labelsize)
    ax.set_xlabel("Hours",fontsize=labelsize)
    return

def boxplot_volt(p_demand, dist, node_interest, ax, **kwargs):
    
    # keyword arguments
    start = kwargs.get("start", 11)
    end = kwargs.get("end", 23)
    shift = kwargs.get("shift", 6)
    ticklabelsize = kwargs.get("tick_labelsize", 30)
    labelsize = kwargs.get("labelsize", 30)
    
    # Initialize data for pandas dataframe
    data = {'voltage':[],'hour':[]}
    
    # Get voltage for individual optimization
    volt_ind = compute_voltage(dist, p_demand)
    
    for t in range(start,end+1):
        hr = f"{(t+shift-1) % 24}:00 - {(t+shift) % 24}:00"
        for n in node_interest:
            data['voltage'].append(volt_ind[n][t])
            data['hour'].append(hr)
    
    # construct pandas dataframe for plot           
    df = pd.DataFrame(data)
    ax = sns.boxplot(x="hour", y="voltage",
                 data=df, color=sns.color_palette("Set2")[1], ax=ax)
    
    ax.tick_params(axis='y',labelsize=ticklabelsize)
    ax.tick_params(axis='x',rotation=45,labelsize=ticklabelsize)
    ax.set_ylabel("Node voltage (p.u.)",fontsize=labelsize)
    ax.set_xlabel("Hours",fontsize=labelsize)
    return




























