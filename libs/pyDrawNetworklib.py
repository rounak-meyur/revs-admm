# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:45:38 2021

Author: Rounak
Description: Functions to create network representations and color graphs based
on their attributes.
"""

from shapely.geometry import Point
import geopandas as gpd

#%% Network Geometries
def DrawNodes(synth_graph,ax,label=['S','T','H'],color='green',size=25,
              alpha=1.0):
    """
    Get the node geometries in the network graph for the specified node label.
    """
    # Get the nodes for the specified label
    if label == []:
        nodelist = list(synth_graph.nodes())
    else:
        nodelist = [n for n in synth_graph.nodes() \
                    if synth_graph.nodes[n]['label']==label \
                        or synth_graph.nodes[n]['label'] in label]
    # Get the dataframe for node and edge geometries
    d = {'nodes':nodelist,
         'geometry':[Point(synth_graph.nodes[n]['cord']) for n in nodelist]}
    df_nodes = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_nodes.plot(ax=ax,color=color,markersize=size,alpha=alpha)
    return

def DrawEdges(synth_graph,ax,label=['P','E','S'],color='black',width=2.0,
              style='solid',alpha=1.0):
    """
    """
    # Get the nodes for the specified label
    if label == []:
        edgelist = list(synth_graph.edges())
    else:
        edgelist = [e for e in synth_graph.edges() \
                    if synth_graph[e[0]][e[1]]['label']==label\
                        or synth_graph[e[0]][e[1]]['label'] in label]
    d = {'edges':edgelist,
         'geometry':[synth_graph.edges[e]['geometry'] for e in edgelist]}
    df_edges = gpd.GeoDataFrame(d, crs="EPSG:4326")
    df_edges.plot(ax=ax,edgecolor=color,linewidth=width,linestyle=style,alpha=alpha)
    return
