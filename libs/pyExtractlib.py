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

def get_home_data(home_filename,ghi_filename,p=0):
    # Extract solar GHI data
    solar_data = get_solar_irradiance(ghi_filename)
    
    # Extract residence device data
    df_homes = pd.read_csv(home_filename)
    np.random.seed(12345)
    home_rawdata = df_homes.set_index('hid').T.to_dict()
    home_data = {h: {"PV":{},"ESS":{},"SL":{},"TCL":{},"FIXED":{}} \
                 for h in home_rawdata}
    rnd = np.random.binomial(n=1,p=p,size=len(home_data))
    for i,h in enumerate(home_rawdata):
        # PV generator and ESS unit
        if rnd[i] == 1:
            tract = home_rawdata[h]["tract"]
            home_data[h]["PV"]["PV1"] = {"rating":0.7,
                                         "solar":solar_data[tract]}
            home_data[h]["ESS"]["ESS1"] = {"modes":5,"rating":1.0,
                                           "capacity":10.0,"initial":0.5}
        
        # Schedulable loads
        if home_rawdata[h]["hasCw"]:
            home_data[h]["SL"]["Laundry"] = {"rating":1.5,"time":2}
        if home_rawdata[h]["hasDw"]:
            home_data[h]["SL"]["Dishwasher"] = {"rating":0.9,"time":2}
        
        # Fixed loads
        home_data[h]["FIXED"]["base"] = [home_rawdata[h]["base_load_"+str(i+1)] \
                                         for i in range(24)]
        home_data[h]["FIXED"]["hvac"] = [home_rawdata[h]["hvac_kwh_"+str(i+1)] \
                                         for i in range(24)]
        home_data[h]["FIXED"]["hoth2o"] = [home_rawdata[h]["hoth2o_kwh_"+str(i+1)] \
                                         for i in range(24)]
    return home_data
    
    
def get_solar_irradiance(filename):
    df_solar = pd.read_csv(filename)
    num_tracts = int(len(df_solar)/24)
    solar = {df_solar['TRACT_FIPS'][t*24] : [df_solar['GHI'][t*24+i] \
                                             for i in range(24)] \
             for t in range(num_tracts)}
    return solar


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