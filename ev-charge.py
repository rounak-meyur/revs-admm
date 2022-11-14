# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 23:31:17 2022

Author: Rounak Meyur

Description: Model EV charging demand
"""

import numpy as np
import matplotlib.pyplot as plt

# P0 = 0
# P_max = 189.0
# t_max = 3.5
# a = 6.9077

def charge(t,P0=0,P_max=189.0,t_max=3.5,a=6.9077):
    return P_max*(1-np.exp(-a*t/t_max)) + P0

time = [i*(1.0/3) for i in range(12)]
p1 = [charge(t,P0=0) for t in time]
p2 = [charge(t,P0=50) for t in time]
p3 = [charge(t,P0=100) for t in time]

fig = plt.figure(figsize=(12,10))
ax = fig.add_subplot(111)

ax.plot(time,p1,color='red',linestyle='dashed',linewidth=2.0)
ax.plot(time,p2,color='blue',linestyle='dashed',linewidth=2.0)
ax.plot(time,p3,color='darkgreen',linestyle='dashed',linewidth=2.0)

ax.set_xlabel("Charging time (in hours)",fontsize=25)
ax.set_ylabel("Charge status (in kWh)",fontsize=25)

ax.set_xlim(0,3.5)
ax.set_ylim(0,210)

ax.hlines(189.0,0.0,3.5,linestyle='dotted',color='black')
ax.vlines([0.3225,0.6737],0.0,210,linestyle='dotted',color='black')