# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:50:32 2021

@author: Rounak Meyur and Swapna Thorve
"""

import sys,os


workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = workpath + "/libs/"
inppath = workpath + "/input/"
figpath = workpath + "/figs/"
outpath = workpath + "/out/"
grbpath = workpath + "/gurobi/"


sys.path.append(libpath)
from pyExtractlib import get_home_data
from pyLoadlib import Load
print("Imported modules")


#%% Main code

homepath = inppath + "VA121_20140723.csv"
solarpath = inppath + "va121_20140723_ghi.csv"

homes = get_home_data(homepath,solarpath)

# Cost profile taken for the off peak plan of AEP
cost = [0.096223]*5 + [0.099686] + [0.170936]*3 + [0.099686]*8 + [0.170936]*3 + [0.099686]*4
# Feed in tarriff rate for small residence owned rooftop solar installations
feed = 0.38


#%% Optimize power usage schedule
P_scheduled = {}
homelist = list(homes.keys())[:20]
for hid in homelist:
    L = Load(24,homes[hid])

    L.set_objective(cost, feed)
    P_scheduled[hid] = L.solve(hid,grbpath)