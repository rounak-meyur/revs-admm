# -*- coding: utf-8 -*-
"""
Created on Thu May  5 09:18:21 2022

@author: rm5nz
"""

import os,sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


workpath = os.getcwd()
libpath = workpath + "/libs/"
figpath = workpath + "/figs/"
inppath = workpath + "/input/"
grbpath = workpath + "/gurobi/"
outpath = workpath + "/out/"

sys.path.append(libpath)
from pyExtractlib import get_home_load


load = get_home_load(inppath+"121-home-load.csv")


#%% Create the load profile
dict_load = {"time":[],"home":[],"has_ev":[],"load":[]}
for h in load:
    for t in range(96):
        dict_load["time"].append(t)
        dict_load["home"].append(h)
        dict_load["load"].append(
            load[h]["fixed"][t] + np.random.normal(0, 0.01, size=1)[0])
        dict_load["has_ev"].append("without EV")

for h in load:
    t_start = np.random.randint(56,72)
    for t in range(96):
        dict_load["time"].append(t)
        dict_load["home"].append(h)
        dict_load["has_ev"].append("with EV")
        if t<t_start or t>t_start+20:
            dict_load["load"].append(
            load[h]["fixed"][t] + np.random.normal(0, 0.01, size=1)[0])
        else:
            dict_load["load"].append(
            load[h]["fixed"][t] + np.random.normal(0, 0.01, size=1)[0] + 2.0)
            

df = pd.DataFrame(dict_load)

#%% Plot the figure
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(111)
ax = sns.lineplot(data=df, x="time",y="load",hue="has_ev",
                  palette=["teal","dodgerblue"],
                 ax=ax,lw=3.0,linestyle='solid',marker='o',markersize=1,
                 err_style='bars',err_kws={'capsize':5.0})


xticks = range(0,96,12)
xtklab = [str((int(x/4)+6)%24)+":00" for x in xticks]
ax.set_xticks(xticks)
ax.set_xticklabels(xtklab)
ax.tick_params(axis='y',labelsize=30)
ax.tick_params(axis='x',labelsize=30)
ax.set_ylabel("Load demand (kW)",fontsize=35)
ax.set_xlabel("Time (HH:MM)",fontsize=35)
ax.legend(ncol=1,markerscale=4.0,loc='upper left',fontsize=35)
ax.grid(visible=True,color='black', linestyle='solid', linewidth=0.3,axis='y')
# ax.set_title("Hourly residential demand in Virginia, USA",fontsize=35)
figpath = "C:/Users/rm5nz/OneDrive - University of Virginia/Pictures/Proposal/"
fig.savefig("{}{}.png".format(figpath,'demand-alter'),
            bbox_inches='tight')