# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 12:46:21 2022

@author: Rounak Meyur
"""

import sys,os
import numpy as np


workpath = os.getcwd()
libpath = workpath + "/libs/"

scratchpath = "/sfs/lustre/bahamut/scratch/rm5nz/ev-schedule"
grbpath = scratchpath + "/gurobi/"
outpath = scratchpath + "/out/"
homepath = scratchpath + "/input/load/121-home-load.csv"
distpath = scratchpath + "/input/osm-primnet/"


sys.path.append(libpath)
from pyExtractlib import get_home_load, GetDistNet
from pySchedEVChargelib import solve_ADMM

print("Imported modules and libraries")

#%% Arguments and inputs
a = sys.argv[1]
r = sys.argv[2]
initial = 0.5
final = 0.9
start = 11
end = 23

#%% Other input parameters
s = 6
sub = 121144
all_homes = get_home_load(homepath,shift=s)
dist = GetDistNet(distpath,sub)
res = [n for n in dist if dist.nodes[n]['label']=='H']
print("\nLoaded network with "+str(len(res))+" residences")

# Area of interest where EV adoption is studied
xmin = -80.37217
ymin = 37.1993
xmax = xmin + 0.010
ymax = ymin + 0.005
res_interest = [n for n in res if (xmin<=dist.nodes[n]['cord'][0]<=xmax) \
                and (ymin<=dist.nodes[n]['cord'][1]<=ymax)]
print("\nArea selected for EV adoption study")


#%% Parameters
np.random.seed(1234)
adoption = float(a)/100.0
rating = float(r)*1e-3
num_choice = int(adoption * len(res_interest))
ev_home = np.random.choice(res_interest,num_choice,replace=False)

homes = {}
for h in res:
    homes[h] = {}
    homes[h]["LOAD"] = [l for l in all_homes[h]["LOAD"]]
    if h in ev_home:
        homes[h]["EV"] = {"rating":rating,"capacity":16.0,
                          "initial":initial,"final":final,
                          "start":start,"end":end}
    else:
        homes[h]["EV"] = {}

# Other input parameters
COST = [1] * 24
vset = 1.03


#%% Distributed algorithm

diff,P_res,P_ev,SOC = solve_ADMM(homes,dist,COST,
                                 grbpath=grbpath,vset=vset,vlow=0.9)


# Save the data in a text file
line1 = "Substation:\t"+str(sub)
line2 = "Substation voltage:\t"+str(vset)
line3 = "Time window start:\t"+str(s)+"a.m."
line4 = "Adoption rate:\t"+str(a)+"%"
line5 = "EV charger rating:\t"+str(r)+"Watts"
line6 = "EV adopters:\t"+' '.join([str(x) for x in ev_home])
line7 = "EV charging time slot start:\t"+str(start+s)+"hours"
line8 = "EV charging time slot period:\t"+str(end-start)+"hours"
line9 = "EV charging:\t"+str(initial*100.0)+"% to "+str(final*100.0)+"%"

data = "\n".join([line1,line2,line3,line4,line5,line6,line7,line8,line9])

# Insert separator
data += "\n#############################################"
data += "\nResidence Usage Profile"
data += "\n#############################################\n"
# Insert Residence usage profile
res_data = '\n'.join([str(h) + ":\t"+' '.join([str(y) for y in P_res[h]]) for h in P_res])
data += res_data

# Insert separator
data += "\n#############################################"
data += "\nEV Charger Usage Profile"
data += "\n#############################################\n"
# Insert EV charger usage profile
ev_data = '\n'.join([str(h) + ":\t"+' '.join([str(z) for z in P_ev[h]]) for h in ev_home])
data += ev_data

# Insert separator
data += "\n#############################################"
data += "\nEV Charger State of Charge Profile"
data += "\n#############################################\n"
# Insert EV charger SOC profile
soc_data = '\n'.join([str(h) + ":\t"+' '.join([str(z) for z in SOC[h]]) for h in ev_home])
data += soc_data

# Insert separator
data += "\n#############################################"
data += "\nEV Convergence over Iterations"
data += "\n#############################################\n"
# Insert convergence result
diff_data = '\n'.join([str(h) + ":\t"+' '.join([str(diff[k+1][h]) \
                                                for k in range(len(diff))]) \
                       for h in ev_home])
data += soc_data

filename = "agentEV-"+str(int(100*adoption))+"-adopt"+str(int(r))+"Watts.txt"
with open(outpath+filename,'w') as f:
    f.write(data)


