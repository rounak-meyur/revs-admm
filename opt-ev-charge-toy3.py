# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:50:32 2021

@author: Rounak Meyur and Swapna Thorve
"""

import sys,os
import numpy as np
import matplotlib.pyplot as plt


workpath = os.getcwd()
libpath = workpath + "/libs/"
figpath = workpath + "/figs/"
outpath = workpath + "/out/"
distpath = workpath + "/input/osm-primnet/"
grbpath = workpath + "/gurobi/"
homepath = workpath + "/input/load/121-home-load.csv"


sys.path.append(libpath)
from pyExtractlib import GetDistNet
from pySchedEVChargelib import compute_Rmat
from pyDrawNetworklib import DrawNodes,DrawEdges
print("Imported modules")

def get_data(datalines):
    dict_data = {}
    for temp in datalines:
        h = int(temp.split('\t')[0][:-1])
        dict_data[h] = [float(x) \
                        for x in temp.split('\t')[1].strip('\n').split(' ')]
    return dict_data

def plot_network(ax,net,path=None,with_secnet=False):
    """
    """
    # Draw nodes
    DrawNodes(net,ax,label='S',color='dodgerblue',size=2000)
    DrawNodes(net,ax,label='T',color='green',size=25)
    DrawNodes(net,ax,label='R',color='black',size=2.0)
    if with_secnet: DrawNodes(net,ax,label='H',color='crimson',size=2.0)
    # Draw edges
    DrawEdges(net,ax,label='P',color='black',width=2.0)
    DrawEdges(net,ax,label='E',color='dodgerblue',width=2.0)
    if with_secnet: DrawEdges(net,ax,label='S',color='crimson',width=1.0)
    ax.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
    if path != None: 
        fig.savefig("{}{}.png".format(path,'-51121-dist'),bbox_inches='tight')
    return ax




#%% Extract results
a = 10
r = 800

prefix = "agentEV-"+str(a)+"-adopt"+str(int(r))+"Watts.txt"
with open(outpath+prefix,'r') as f:
    lines = f.readlines()

sub = int(lines[0].split('\t')[1])
vset = float(lines[1].split('\t')[1])
s = 6
ev_home = [int(x) for x in lines[5].split('\t')[1].split(' ')]
T = 24

# Get usage profile
sepind = [i+1 for i,l in enumerate(lines) if l.strip("\n").endswith("##")]

res_lines = lines[sepind[1]:sepind[2]-1]
ev_lines = lines[sepind[3]:sepind[4]-1]
soc_lines = lines[sepind[5]:sepind[6]-1]
diff_lines = lines[sepind[7]:len(lines)]

P_res = get_data(res_lines)
P_ev = get_data(ev_lines)
SOC = get_data(soc_lines)
diff = get_data(diff_lines)




#%% Get the data

# all_homes = get_home_load(homepath,shift=s)
dist = GetDistNet(distpath,sub)

res = [n for n in dist if dist.nodes[n]['label']=='H']

# Area of interest where EV adoption is studied
xmin = -80.37217
ymin = 37.1993
xmax = xmin + 0.010
ymax = ymin + 0.005
res_interest = [n for n in res \
                if (xmin<=dist.nodes[n]['cord'][0]<=xmax) \
                    and (ymin<=dist.nodes[n]['cord'][1]<=ymax)]

R = compute_Rmat(dist)
nodelist = [n for n in dist if dist.nodes[n]['label']!='S']
vset=1.0


#%% Multiple voltage profiles
import matplotlib.patches as patches


colors = ['pink','lightblue','lightgreen','lightsalmon','lightyellow',
          'orchid','bisque','aquamarine','slateblue']
labels = [(c+s)%24 for c in range(11,24)]


for a in [10,20,30,40,50,60]:
    fig = plt.figure(figsize=(60,20))
    ax1 = fig.add_subplot(1,2,2)
    ax2 = fig.add_subplot(1,2,1)
    ax2 = plot_network(ax2,dist,with_secnet=True)
    ax2.set_title("EV adoption percentage: "+str(a)+"%",fontsize=50)

    # Create a Rectangle patch
    rect = patches.Rectangle((-80.37217, 37.1993), 0.01, 0.005, linewidth=1, edgecolor='blue', facecolor='none')
    # Add the patch to the Axes to denote the locality
    ax2.add_patch(rect)
    # Add the nodes where EV adoption
    prefix = "agentEV-"+str(a)+"-adopt800Watts.txt"
    with open(outpath+prefix,'r') as f:
        lines = f.readlines()
    ev_homes = [int(x) for x in lines[5].split('\t')[1].split(' ')]
    x_int = [dist.nodes[n]['cord'][0] for n in ev_homes]
    y_int = [dist.nodes[n]['cord'][1] for n in ev_homes]
    ax2.scatter(x_int,y_int,s=100.0,c='b')
    
    
    leghands = []
    for b,r in enumerate([800, 1000, 1200, 1600]):
        # Extract data
        prefix = "agentEV-"+str(a)+"-adopt"+str(int(r))+"Watts.txt"
        with open(outpath+prefix,'r') as f:
            lines = f.readlines()
        
        sepind = [i+1 for i,l in enumerate(lines) if l.strip("\n").endswith("##")]
        res_lines = lines[sepind[1]:sepind[2]-1]
        diff_lines = lines[sepind[7]:len(lines)]
        P_res = get_data(res_lines)
        diff = get_data(diff_lines)
        
        # Compute voltages at nodes
        P = np.zeros(shape=(len(nodelist),T))
        Z = np.ones(shape=(len(nodelist),T)) * vset
        for i,n in enumerate(nodelist):
            if n in res:
                P[i,:] = np.array(P_res[n])
        
        V = np.sqrt(Z - R@P)
        volt = {h:V[i,:].tolist() for i,h in enumerate(nodelist) if h in nodelist}
        volt_data = [[volt[n][t] for n in res_interest] for t in range(11,24)]
        
        # Add the plot to figure
        bplot = ax1.boxplot(volt_data,patch_artist=True,labels=labels)
        ax1.tick_params(axis='y',labelsize=50)
        ax1.tick_params(axis='x',labelsize=50)
        ax1.set_ylabel("Node Voltage",fontsize=50)
        ax1.set_xlabel("Hours",fontsize=50)
        for t in range(len(labels)):
            bplot['boxes'][t].set_facecolor(colors[b])
        
        leghands.append(patches.Rectangle((0,0),1,1, facecolor=colors[b],
                               label=str(r)+" Watts"))
        ax1.legend(handles=leghands,loc='best',ncol=2,prop={'size': 40})
        # # Plot the convergence
        # num_iter = 15
        # xtix = range(1,num_iter+1)
        # ax4 = fig4.add_subplot(3,3,l+1)
        # for h in ev_home:
        #     ax4.plot(xtix,[diff[h][k] for k in range(num_iter)],
        #               label="Home "+str(h))
        # ax4.set_ylabel("Difference",fontsize=25)
        # ax4.set_xlabel("Iterations",fontsize=25)
        # ax4.legend(loc='best',ncol=2,prop={'size': 25})
        # ax4.set_xticks(list(range(0,num_iter+1,5)))
        # ax4.tick_params(axis='y',labelsize=25)
        # ax4.tick_params(axis='x',labelsize=25)
    
    fig.savefig("{}{}.png".format(figpath,str(sub)+"-adoption-"+str(a)),
                bbox_inches='tight')
        
        
    
    









