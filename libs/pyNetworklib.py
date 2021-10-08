# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 16:10:51 2021

@author: Rounak Meyur
"""

import networkx as nx
import numpy as np
from math import log



def powerflow(graph):
    """
    Checks power flow solution and save dictionary of voltages.
    """
    A = nx.incidence_matrix(graph,nodelist=list(graph.nodes()),
                            edgelist=list(graph.edges()),oriented=True).toarray()
    
    node_ind = [i for i,node in enumerate(graph.nodes()) \
                if graph.nodes[node]['label'] != 'S']
    nodelist = [node for node in list(graph.nodes()) \
                if graph.nodes[node]['label'] != 'S']
    edgelist = [edge for edge in list(graph.edges())]
    
    # Resistance data
    edge_r = []
    for e in graph.edges:
        try:
            edge_r.append(1.0/graph.edges[e]['r'])
        except:
            edge_r.append(1.0/1e-14)
    R = np.diag(edge_r)
    G = np.matmul(np.matmul(A,R),A.T)[node_ind,:][:,node_ind]
    
    for t in range(24):
        p = np.array([graph.nodes[n]['load'][t] for n in nodelist])
        dv = np.matmul(np.linalg.inv(G),p)
        v = {h:1.0-v[i] for i,h in enumerate(nodelist)}
    return