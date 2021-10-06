# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:47:41 2021

Authors: Rounak Meyur and Swapna Thorve

Description: Functions to extract residence data
"""

import pandas as pd
import numpy as np

#%% Functions to extract data

def get_home_data(home_filename,ghi_filename):
    # Extract solar GHI data
    solar_data = get_solar_irradiance(ghi_filename)
    
    # Extract residence device data
    df_homes = pd.read_csv(home_filename)
    np.random.seed(12345)
    home_rawdata = df_homes.set_index('hid').T.to_dict()
    home_data = {h: {"PV":{},"ESS":{},"SL":{},"TCL":{},"FIXED":{}} \
                 for h in home_rawdata}
    for h in home_rawdata:
        # PV generator and ESS unit
        if np.random.binomial(n=1,p=0.2) == 1:
            tract = home_rawdata[h]["tract"]
            home_data[h]["PV"]["PV1"] = {"rating":0.7,
                                         "solar":solar_data[tract]}
            home_data[h]["ESS"]["ESS1"] = {"modes":5,"rating":1.0,
                                           "capacity":10.0,"initial":0.5}
        
        # Schedulable loads
        if home_rawdata[h]["hasCw"]:
            home_data[h]["SL"]["Laundry"] = {"rating":1.5,"time":5}
        if home_rawdata[h]["hasDw"]:
            home_data[h]["SL"]["Dishwasher"] = {"rating":0.9,"time":5}
        
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