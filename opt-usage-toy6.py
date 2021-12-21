# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:50:32 2021

@author: Rounak Meyur and Swapna Thorve
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt

workpath = os.getcwd()
rootpath = os.path.dirname(workpath)
libpath = workpath + "/libs/"
inppath = workpath + "/input/"
figpath = workpath + "/figs/"
outpath = workpath + "/out/"
distpath = rootpath + "/synthetic-distribution/primnet/out/osm-primnet/"
grbpath = workpath + "/gurobi/"
homepath = inppath + "VA121_20140723.csv"


sys.path.append(libpath)
# from pySchedLoadlib import Central
print("Imported modules")




COST = [1]*6
ALLOT = {1:4,2:5,3:3,4:1,5:2,6:3}


#%% Distributed algorithm
from pySchedLoadlib import Agent,Authority

homelist = [1,2,3,4,5,6]



P_est = {0:{h:[0]*len(COST) for h in homelist}}
P_sch = {0:{h:[0]*len(COST) for h in homelist}}
G = {0:{h:[0]*len(COST) for h in homelist}}
S = {}

kappa = 5.0
iter_max = 50
error = [1]

k = 0
eps = 1
while(k <= iter_max):
    # solve utility level problem to get estimate
    U_obj = Authority(P_est[k],P_sch[k],G[k],kappa=kappa,max_allot=3)
    U_obj.solve(grbpath)
    P_est[k+1] = U_obj.g_opt
    
    
    # solve individual agent level problem
    P_sch[k+1] = {}
    S[k+1] = {}
    for h in homelist:
        H_obj = Agent(COST,ALLOT[h],P_est[k][h],P_sch[k][h],G[k][h],kappa=kappa)
        H_obj.solve(grbpath)
        P_sch[k+1][h] = H_obj.g_opt
    
    
    # update dual variables
    
    G[k+1] = {}
    diff = []
    for h in homelist:
        check = [(P_est[k+1][h][t] - P_sch[k+1][h][t]) for t in range(len(COST))]
        G[k+1][h] = [G[k][h][t] + (kappa/2) * check[t]for t in range(len(COST))]
        diff.append(np.linalg.norm(np.array(check))/len(COST))
    eps = np.linalg.norm(np.array(diff))/len(diff)
    error.append(eps)
    
    k = k + 1



#%% Plot incentive evolution
xarray = np.linspace(0,len(COST)+1,len(COST)+1)
iterations = k
l = iterations-5




# fig2 = plt.figure(figsize=(20,16))
# for i,h in enumerate(homelist):
#     ax2 = fig2.add_subplot(6,1,i+1)
#     for m in range(l,iterations+1):
#         ax2.step(xarray,[0]+P_sch[m][h],label="iter="+str(m))
#         ax2.legend(ncol=5,prop={'size': 15})
#     ax2.set_ylabel("Total load (kW)",fontsize=15)

fig2 = plt.figure(figsize=(20,16))
for i,h in enumerate(homelist):
    ax2 = fig2.add_subplot(6,1,i+1)
    ax2.step(xarray,[0]+P_sch[k][h])
    ax2.set_ylabel("Total load (kW)",fontsize=15)



fig3 = plt.figure(figsize=(20,16))
for i,h in enumerate(homelist):
    ax3 = fig3.add_subplot(6,1,i+1)
    for m in range(l,iterations+1):
        ax3.step(xarray,[0]+G[m][h],label="iter="+str(m))
        ax3.legend(ncol=5,prop={'size': 15})
    ax3.set_ylabel("Incentive signal",fontsize=15)


fig4 = plt.figure(figsize=(20,16))
ax4 = fig4.add_subplot(1,1,1)
ax4.plot(range(iterations),error[1:])
ax4.set_ylabel("Difference (pu)",fontsize=15)


