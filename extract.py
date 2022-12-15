# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:47:41 2021

Authors: Rounak Meyur and Swapna Thorve

Description: Functions to extract residence data
"""

import os
import pandas as pd
import numpy as np
import networkx as nx

#%% Functions to extract data
def GetTariff(path, region, shift):
    if not os.path.exists(f"{path}/{region}-tariff.txt"):
        raise ValueError(f"{path}/{region}-tariff.txt doesn't exist!")
    
    with open(f"{path}/{region}-tariff.txt", "r") as f:
        tariff = [float(x) for x in f.readline().split(" ")]
    
    tariff = np.roll(tariff, -shift).tolist()
    return tariff

def GetHomeLoad(path, region_list, shift):
    home_data = {}
    if not isinstance(region_list, (list,tuple)):
        region_list = [region_list]
    
    for reg in region_list:
        if not os.path.exists(f"{path}/{reg}-home-load.csv"):
            raise ValueError(f"{path}/{reg}-home-load.csv doesn't exist!")
            
        else:
            df_homes = pd.read_csv(f"{path}/{reg}-home-load.csv")
            home_rawdata = df_homes.set_index('hid').T.to_dict()
            
            
            for h in home_rawdata:
                net_load = [1e-3*home_rawdata[h]["hour"+str(i+1)] \
                            for i in range(24)]
                home_data[h] = np.roll(net_load,-shift).tolist()
        
    return home_data


def GetDistNet(path,code):
    """
    Read the txt file containing the edgelist of the generated synthetic network and
    generates the corresponding networkx graph. The graph has the necessary node and
    edge attributes.
    
    Inputs:
        path: name of the directory
        code: substation ID or list of substation IDs
        
    Output:
        graph: networkx graph
        node attributes of graph:
            cord: longitude,latitude information of each node
            label: 'H' for home, 'T' for transformer, 'R' for road node, 
                    'S' for subs
            voltage: node voltage in pu
        edge attributes of graph:
            label: 'P' for primary, 'S' for secondary, 'E' for feeder lines
            r: resistance of edge
            x: reactance of edge
            geometry: shapely geometry of edge
            geo_length: length of edge in meters
            flow: power flowing in kVA through edge
    """
    if isinstance(code,list):
        graph = nx.Graph()
        for c in code:
            g = nx.read_gpickle(f"{path}/{c}-dist-net.gpickle")
            graph = nx.compose(graph,g)
    else:
        graph = nx.read_gpickle(f"{path}/{code}-dist-net.gpickle")
    return graph

def GetCommunity(filename, com_index):
    if not os.path.exists(f"{filename}"):
        raise ValueError(f"{filename} doesn't exist!")
    with open(f"{filename}",'r') as f:
        lines = f.readlines()
    com = [int(x) for x in lines[int(com_index)-1].strip('\n').split(' ')]
    return com


def get_homes_ev_param(homes, dist, ev_homes, 
                       rating, capacity, initial, start, end):
    
    # Change input data type of EV charger rating
    if not isinstance(rating, dict):
        rating = {h:rating for h in ev_homes}
    
    # Change input data type of EV charge capacity
    if not isinstance(capacity, dict):
        capacity = {h:capacity for h in ev_homes}
        
    # Change input data type of EV initial charge
    if not isinstance(initial, dict):
        initial = {h:initial for h in ev_homes}
    
    # Change input data type of EV charging start time
    if not isinstance(start, dict):
        start = {h:start for h in ev_homes}
    
    # Change input data type of EV charging end time
    if not isinstance(end, dict):
        end = {h:end for h in ev_homes}
    
    
    # Get residences in the network
    res = [n for n in dist if dist.nodes[n]['label']=='H']
    
    # Get dictionary of homes with EV parameters
    home_params = {h:{} for h in res}
    for h in res:
        home_params[h]["LOAD"] = [l for l in homes[h]]
        if h in ev_homes:
            home_params[h]["EV"] = {
                "rating":rating[h],
                "capacity":float(capacity[h]),
                "initial":initial[h],
                "start":start[h],
                "end":end[h]
                }
        else:
            home_params[h]["EV"] = {}
    return home_params

def combine_result(P_res, P_ev, SOC, ev_homes, diff=None):
    
    data = ""
    
    # Insert separator
    data += "\n#############################################"
    data += "\nResidence Usage Profile"
    data += "\n#############################################\n"
    # Insert Residence usage profile
    res_data = '\n'.join([str(h) + ":\t"+' '.join([str(y) for y in P_res[h]]) \
                          for h in P_res])
    data += res_data
    
    # Insert separator
    data += "\n#############################################"
    data += "\nEV Charger Usage Profile"
    data += "\n#############################################\n"
    # Insert EV charger usage profile
    ev_data = '\n'.join([str(h) + ":\t"+' '.join([str(z) for z in P_ev[h]]) \
                         for h in ev_homes])
    data += ev_data
    
    # Insert separator
    data += "\n#############################################"
    data += "\nEV Charger State of Charge Profile"
    data += "\n#############################################\n"
    # Insert EV charger SOC profile
    soc_data = '\n'.join([str(h) + ":\t"+' '.join([str(z) for z in SOC[h]]) \
                          for h in ev_homes])
    data += soc_data
    
    if diff:
    # insert separator
        data += "\n#############################################"
        data += "\nEV Convergence over Iterations"
        data += "\n#############################################\n"
        # Insert convergence result
        diff_data = '\n'.join([str(h) + ":\t"+' '.join([str(diff[k+1][h]) \
                                                        for k in range(len(diff))]) \
                               for h in ev_homes])
        data += diff_data
    return data