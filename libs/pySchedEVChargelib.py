# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 10:27:37 2022

@author: rm5nz
"""

import sys
import gurobipy as grb
import numpy as np
import networkx as nx

def compute_Rmat(graph):
    A = nx.incidence_matrix(graph,nodelist=list(graph.nodes()),
                            edgelist=list(graph.edges()),oriented=True).toarray()
    node_ind = [i for i,node in enumerate(graph.nodes()) \
                if graph.nodes[node]['label'] != 'S']
    
    # Resistance data
    F = np.linalg.inv(A[node_ind,:].T)
    D = np.diag([graph.edges[e]['r'] for e in graph.edges])
    return 2*F@D@(F.T)


def mycallback(model, where):
    if where == grb.GRB.Callback.MIP:
        # General MIP callback
        objbst = model.cbGet(grb.GRB.Callback.MIP_OBJBST)
        objbnd = model.cbGet(grb.GRB.Callback.MIP_OBJBND)
        time = model.cbGet(grb.GRB.Callback.RUNTIME)
        if (time>300 and abs(objbst - objbnd) < 0.05 * (1.0 + abs(objbst))):
            print('Stop early - 5% gap achieved')
            model.terminate()
        elif(time>60 and abs(objbst - objbnd) < 0.01 * (1.0 + abs(objbst))):
            print('Stop early - 1% gap achieved')
            model.terminate()
    return


class Home:
    def __init__(self,cost,homedata,p_est,p_sch,gamma,kappa = 5.0):
        self.c = cost
        self.T = len(cost)
        self.data = homedata
        self.p = {}
        self.g = {}
        self.s = {}
        
        self.model = grb.Model(name="Get Optimal Schedule")
        self.model.ModelSense = grb.GRB.MINIMIZE
        self.add_EV()
        self.netload_var()
        self.set_objective(p_est,p_sch,gamma,kappa=kappa)
        return
    
    def netload_var(self):
        for t in range(self.T):
            self.g[t] = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                          name="g_{0}".format(t))
            self.model.addConstr(
                self.g[t] == self.p[t] + self.data["LOAD"][t])
        return
    
    def add_EV(self):
        e = {}
        EV_data = self.data["EV"]
        prate = EV_data['rating']
        qcap = EV_data['capacity']
        init = EV_data['initial']
        final = EV_data['final']
        start = EV_data['start']
        end = EV_data['end']
            
        for t in range(self.T):
            # Add the variables
            self.s[t] = self.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=init, 
                                          ub=1.0,name="s_{0}".format(t))
            e[t] = self.model.addVar(vtype=grb.GRB.BINARY,
                                         name="e_{0}".format(t))
            self.p[t] = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                           name="p_{0}".format(t))
        
            # Add the constraints
            self.model.addConstr(self.p[t] == e[t]*prate)
            if t == 0:
                self.model.addConstr(self.s[t] == init)
            else:
                self.model.addConstr(self.s[t] == self.s[t-1] + (self.p[t]/qcap))
            if t >= end:
                self.model.addConstr(self.s[t] >= final)
            if (t<=start) or (t>end):
                self.model.addConstr(e[t] == 0)
        return
    
    def set_objective(self,p_util,p_res,gamma,kappa=5.0):
        # main objective
        obj1 = grb.quicksum([(self.c[t]*self.g[t]) for t in range(self.T)])
        
        # admm penalty
        obj2 = (kappa/2.0) * grb.quicksum([self.g[t] * self.g[t] for t in range(self.T)])
        a = [gamma[t] + (kappa/2.0)*(p_util[t] + p_res[t])\
             for t in range(self.T)]
        obj3 = grb.quicksum([self.g[t] * a[t] for t in range(self.T)])
        
        # total objective
        self.model.setObjective(obj1+obj2-obj3)
        return
    
    def solve(self,grbpath):
        # Write the LP problem
        self.model.write(grbpath+"ev-schedule-agent.lp")
        
        # Set up solver settings
        grb.setParam('OutputFlag', 0)
        grb.setParam('Heuristics', 0)
        
        # Open log file
        logfile = open(grbpath+'gurobi-ev-agent.log', 'w')
        
        # Pass data into my callback function
        self.model._lastiter = -grb.GRB.INFINITY
        self.model._lastnode = -grb.GRB.INFINITY
        self.model._logfile = logfile
        self.model._vars = self.model.getVars()
        
        # Solve model and capture solution information
        self.model.optimize(mycallback)
        
        # Close log file
        logfile.close()
        if self.model.SolCount == 0:
            print('No solution found, optimization status = %d' % self.model.Status)
            sys.exit(0)
        else:
            self.p_opt = [self.p[t].getAttr("x") for t in range(self.T)]
            self.s_opt = [self.s[t].getAttr("x") for t in range(self.T)]
            self.g_opt = [self.g[t].getAttr("x") for t in range(self.T)]
            return
        

class Utility:
    def __init__(self,graph,P_util,P_sch,Gamma,kappa=5.0,low=0.95,high=1.05):
        self.nodes = [n for n in graph if graph.nodes[n]['label'] != 'S']
        self.res = [n for n in graph if graph.nodes[n]['label'] == 'H']
        self.N = len(self.nodes)
        self.T = len(Gamma[self.res[0]])
        
        self.model = grb.Model(name="Get Optimal Utility Estimated Schedule")
        self.model.ModelSense = grb.GRB.MINIMIZE
        self.variables()
        self.network(graph,vmin=low,vmax=high)
        self.set_objective(P_util,P_sch,Gamma,kappa=kappa)
        return
    
    def variables(self):
        self.g = {(n,t):self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                          name="g_{0}_{1}".format(n,t)) \
                  for n in self.res for t in range(self.T)}
        return
    
    def network(self,graph,vmin=0.95,vmax=1.05):
        R = compute_Rmat(graph)
        vlow = (vmin*vmin - 1)
        vhigh = (vmax*vmax - 1)
        
        # Indices of residence nodes in the R matrix
        resind = [i for i,n in enumerate(self.nodes) if n in self.res]
        
        for i in resind:
            for t in range(self.T):
                self.model.addConstr(
                    -grb.quicksum([R[i,j]*self.g[(self.nodes[j],t)] \
                                   for j in resind]) <= vhigh)
                self.model.addConstr(
                    -grb.quicksum([R[i,j]*self.g[(self.nodes[j],t)] \
                                  for j in resind]) >= vlow)
        return
    
    def set_objective(self,p_util,p_res,gamma,kappa=5.0):
        # admm penalty
        obj1 = (kappa/2.0) * grb.quicksum([self.g[(n,t)] * self.g[(n,t)] \
                                           for n in self.res for t in range(self.T)])
        obj2 = 0
        for n in self.res:
            a = [gamma[n][t] - (kappa/2.0)*(p_util[n][t] + p_res[n][t])\
                 for t in range(self.T)]
            obj2 += grb.quicksum([self.g[(n,t)] * a[t] \
                                                for t in range(self.T)])
        
        # total objective
        self.model.setObjective(obj1+obj2)
        return
    
    def solve(self,grbpath):
        # Write the LP problem
        self.model.write(grbpath+"load-schedule-utility.lp")
        
        # Set up solver settings
        grb.setParam('OutputFlag', 0)
        grb.setParam('Heuristics', 0)
        
        # Open log file
        logfile = open(grbpath+'gurobi-utility.log', 'w')
        
        # Pass data into my callback function
        self.model._lastiter = -grb.GRB.INFINITY
        self.model._lastnode = -grb.GRB.INFINITY
        self.model._logfile = logfile
        self.model._vars = self.model.getVars()
        
        # Solve model and capture solution information
        self.model.optimize(mycallback)
        
        # Close log file
        logfile.close()
        if self.model.SolCount == 0:
            print('No solution found, optimization status = %d' % self.model.Status)
            sys.exit(0)
        else:
            self.g_opt = {h: [self.g[(h,t)].getAttr("x") \
                              for t in range(self.T)] for h in self.res}
            return