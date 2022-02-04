# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 12:37:23 2022

@author: Rounak Meyur
"""

import sys,os
import numpy as np


workpath = os.getcwd()
libpath = workpath + "/libs/"

scratchpath = "/sfs/lustre/bahamut/scratch/rm5nz/ev-schedule"
grbpath = scratchpath + "/gurobi/"
outpath = scratchpath + "/out/"
compath = scratchpath + "/input/"
homepath = scratchpath + "/input/load/121-home-load.csv"
distpath = scratchpath + "/input/osm-primnet/"


sys.path.append(libpath)
from pyExtractlib import get_home_load,GetDistNet
from pySchedEVChargelib import Residence

print("Imported modules and libraries")

#%% Arguments and inputs
comp_ind = sys.argv[1]
a = sys.argv[2]
r = sys.argv[3]
capacity = 20
initial = 0.2
start = 11
end = 23

#%% Other input parameters
s = 6
sub = 121144
all_homes = get_home_load(homepath,shift=s)
dist = GetDistNet(distpath,sub)
res = [n for n in dist if dist.nodes[n]['label']=='H']

with open(compath+str(sub)+"-com.txt",'r') as f:
    lines = f.readlines()
res_interest = [int(x) for x in lines[int(comp_ind)-1].strip('\n').split(' ')]


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
        homes[h]["EV"] = {"rating":rating,"capacity":float(capacity),
                          "initial":initial,"start":start,"end":end}
    else:
        homes[h]["EV"] = {}

# Other input parameters from DOMINION Energy
COST = [0.078660]*5 + [0.095111]*10 + [0.214357]*3 + [0.095111]*6
COST = np.roll(COST,-s).tolist()
vset = 1.03


#%% Run simulation with no distributed scheduling
P_res = {}
P_ev = {}
SOC = {}
for h in res:
    H_obj = Residence(COST,homes[h])
    H_obj.solve(grbpath)
    P_res[h] = H_obj.g_opt
    SOC[h] = H_obj.s_opt
    P_ev[h] = H_obj.p_opt


#%% Save the data
# Save the data in a text file
line1 = "Substation:\t"+str(sub)
line2 = "Substation voltage:\t"+str(vset)
line3 = "Time window start:\t"+str(s)+"a.m."
line4 = "Adoption rate:\t"+str(a)+"%"
line5 = "EV charger rating:\t"+str(r)+"Watts"
line6 = "EV adopters:\t"+' '.join([str(x) for x in ev_home])
line7 = "EV charging time slot start:\t"+str(start+s)+"hours"
line8 = "EV charging time slot period:\t"+str(end-start)+"hours"
line9 = "EV initial charge:\t"+str(initial*100.0)+"%"

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


filename = "indEV-"+str(a)+"-adopt"+str(r)+"Watts.txt"
with open(outpath+str(sub)+"-com-"+str(comp_ind)+"/"+filename,'w') as f:
    f.write(data)














