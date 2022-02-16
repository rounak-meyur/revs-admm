# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:47:41 2021

Authors: Rounak Meyur and Swapna Thorve

Description: Functions to extract residence data
"""

import pandas as pd
import numpy as np
import networkx as nx

#%% Functions to extract data
def get_home_load(home_filename,shift=6):
    # Extract residence device data
    df_homes = pd.read_csv(home_filename)
    home_rawdata = df_homes.set_index('hid').T.to_dict()
    home_data = {h: {} for h in home_rawdata}
    
    for h in home_rawdata:
        net_load = [1e-3*home_rawdata[h]["hour"+str(i+1)] for i in range(24)]
        home_data[h]["LOAD"] = np.roll(net_load,-shift).tolist()
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
    if type(code) == list:
        graph = nx.Graph()
        for c in code:
            g = nx.read_gpickle(path+str(c)+'-dist-net.gpickle')
            graph = nx.compose(graph,g)
    else:
        graph = nx.read_gpickle(path+str(code)+'-dist-net.gpickle')
    return graph