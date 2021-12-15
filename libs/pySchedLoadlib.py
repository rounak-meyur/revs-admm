# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 10:16:57 2021

Author: Swapna Thorve and Rounak Meyur

Description: Residence Load Device models
"""

import sys
import gurobipy as grb
import networkx as nx
import numpy as np

def compute_Rmat(graph):
    A = nx.incidence_matrix(graph,nodelist=list(graph.nodes()),
                            edgelist=list(graph.edges()),oriented=True).toarray()
    node_ind = [i for i,node in enumerate(graph.nodes()) \
                if graph.nodes[node]['label'] != 'S']
    
    # Resistance data
    F = np.linalg.inv(A[node_ind,:].T)
    D = np.diag([graph.edges[e]['r'] for e in graph.edges])
    return 2*F@D@(F.T)

#%% Function for callback
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

#%% Functions for optimization problem

class Central:
    def __init__(self,homedata,graph,cost):
        self.model = grb.Model(name="Get Optimal Schedule")
        self.model.ModelSense = grb.GRB.MINIMIZE
        self.p = {}
        self.g = {}
        self.a = {}
        self.data = homedata
        self.w = len(cost)
        
        # Add all load devices
        for h in homedata:
            self.add_SL(h)
            self.add_fixed(h)
        
        # Add network constraints and incentive
        self.network(graph,cost)
        self.set_objective(graph, cost)
        
        return
    
    def add_SL(self,h):
        u = {}
        SL_data = self.data[h]["SL"]
        for d in SL_data:
            # Schedulable Load parameters
            prate = SL_data[d]["rating"]
            tschd = SL_data[d]["time"]
            for t in range(self.w):
                # Variables
                u[(h,d,t)] = self.model.addVar(vtype=grb.GRB.BINARY,
                                name="u_{0}_{1}_{2}".format(h,d,t))
                self.p[(h,d,t)] = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                name="p_{0}_{1}_{2}".format(h,d,t))
                
            # Constraints
            self.model.addConstr(
                grb.quicksum([u[(h,d,t)] for t in range(self.w)]) == tschd)
            for t in range(self.w):
                self.model.addConstr(self.p[(h,d,t)] == prate*u[(h,d,t)])
            for k in range(self.w-tschd):
                if k==0:
                    self.model.addConstr(
                        grb.quicksum([u[(h,d,t)] \
                                      for t in range(k,k+tschd)]) >= \
                            tschd*u[(h,d,k)])
                else:
                    self.model.addConstr(
                        grb.quicksum([u[(h,d,t)] \
                                      for t in range(k,k+tschd)]) >= \
                            tschd*(u[(h,d,k)]-u[(h,d,k-1)]))
        return
    
    def add_fixed(self,h):
        fixed_data = self.data[h]["FIXED"]
        for d in fixed_data:
            for t in range(self.w):
                self.p[(h,d,t)] = fixed_data[d][t]
        return
    
    def network(self,graph,c0,vmin=0.95,vmax=1.05):
        R = compute_Rmat(graph)
        vlow = (vmin*vmin - 1)
        vhigh = (vmax*vmax - 1)
        nodelist = [n for n in graph.nodes() if graph.nodes[n]['label'] != 'S']
        for n in nodelist:
            for t in range(self.w):
                self.g[(n,t)] = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                            name="g_{0}_{1}".format(n,t))
                if graph.nodes[n]['label']=='H':
                    devices = [d for dtype in self.data[n] if dtype != "ORG" \
                                for d in self.data[n][dtype]]
                    self.model.addConstr(
                        self.g[(n,t)] == grb.quicksum([self.p[(n,d,t)] \
                                                        for d in devices]))
                else:
                    self.model.addConstr(self.g[(n,t)] == 0)
        
        # Add voltage constraints
        for i,_ in enumerate(nodelist):
            for t in range(self.w):
                self.model.addConstr(
                    -grb.quicksum([R[i,j]*self.g[(m,t)] \
                                  for j,m in enumerate(nodelist)]) <= vhigh)
                self.model.addConstr(
                    -grb.quicksum([R[i,j]*self.g[(m,t)] \
                                  for j,m in enumerate(nodelist)]) >= vlow)
        
        for n in nodelist:
            for t in range(self.w):
                self.a[(n,t)] = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                    name="a_{0}_{1}".format(n,t))
                self.model.addConstr(self.a[(n,t)] >= 0)
                self.model.addConstr(self.a[(n,t)] <= c0[t]*self.g[(n,t)])
        return
    
    def set_objective(self,graph,c0):
        nodelist = [n for n in graph.nodes() if graph.nodes[n]['label'] != 'S']
        obj = grb.quicksum([(c0[t]*self.g[(n,t)]) - self.a[(n,t)] \
                            for n in nodelist for t in range(self.w)])
        self.model.setObjective(obj)
        return
    
    def solve(self,grbpath):
        # Write the LP problem
        self.model.write(grbpath+"load-schedule-central.lp")
        
        # Set up solver settings
        grb.setParam('OutputFlag', 0)
        grb.setParam('Heuristics', 0)
        
        # Open log file
        logfile = open(grbpath+'gurobi-central.log', 'w')
        
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
            # Return the device schedules
            fixed = ['base', 'hvac', 'hoth2o']
            self.p_sch = {}
            self.g_opt = {}
            self.a_opt = {}
            for h in self.data:
                devices = [d for dtype in self.data[h] if dtype != "ORG" \
                            for d in self.data[h][dtype] if d not in fixed]
                p_opt = {(h,d,t): self.p[(h,d,t)].getAttr("x") for d in devices \
                          for t in range(self.w)}
            
                # Store optimal schedule in the attribute
                self.p_sch[h] = {d: [p_opt[(h,d,t)] for t in range(self.w)] for d in devices}
                self.g_opt[h] = [self.g[(h,t)].getAttr("x") for t in range(self.w)]
                self.a_opt[h] = [self.a[(h,t)].getAttr("x") for t in range(self.w)]
            return
        
    
    
class Simple:
    def __init__(self,graph,cost):
        self.c = cost
        self.T = len(cost)
        self.nodes = [n for n in graph if graph.nodes[n]['label'] != 'S']
        self.res = [n for n in graph if graph.nodes[n]['label'] == 'H']
        self.N = len(self.nodes)
        
        self.model = grb.Model(name="Get Optimal Schedule")
        self.model.ModelSense = grb.GRB.MINIMIZE
        self.variables()
        self.network(graph)
        self.set_objective()
        return
    
    def variables(self):
        self.g = {(n,t):self.model.addVar(vtype=grb.GRB.BINARY,
                                          name="g_{0}_{1}".format(n,t)) \
                  for n in self.nodes for t in range(self.T)}
        self.a = {(n,t):self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                          name="a_{0}_{1}".format(n,t)) \
                  for n in self.nodes for t in range(self.T)}
        
        # Add constraints for the variables
        for n in self.nodes:
            if n in self.res:
                self.model.addConstr(
                    grb.quicksum([self.g[(n,t)] for t in range(self.T)]) == 1)
            else:
                for t in range(self.T):
                    self.model.addConstr(self.g[(n,t)] == 0)
            for t in range(self.T):
                self.model.addConstr(self.a[(n,t)] >= 0)
                self.model.addConstr(self.a[(n,t)] <= self.c[t]*self.g[(n,t)])
        return
    
    def network(self,graph,vmin=0.95,vmax=1.05):
        R = compute_Rmat(graph)
        vlow = (vmin*vmin - 1)
        vhigh = (vmax*vmax - 1)
        
        for i,_ in enumerate(self.nodes):
            for t in range(self.T):
                self.model.addConstr(
                    -grb.quicksum([R[i,j]*self.g[(m,t)] \
                                  for j,m in enumerate(self.nodes)]) <= vhigh)
                self.model.addConstr(
                    -grb.quicksum([R[i,j]*self.g[(m,t)] \
                                  for j,m in enumerate(self.nodes)]) >= vlow)
        return
    
    def set_objective(self):
        obj = grb.quicksum([(self.c[t]*self.g[(n,t)]) - self.a[(n,t)] \
                            for n in self.nodes for t in range(self.T)])
        self.model.setObjective(obj)
        return
    
    def solve(self,grbpath):
        # Write the LP problem
        self.model.write(grbpath+"load-schedule-simple.lp")
        
        # Set up solver settings
        grb.setParam('OutputFlag', 0)
        grb.setParam('Heuristics', 0)
        
        # Open log file
        logfile = open(grbpath+'gurobi-simple.log', 'w')
        
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
            self.a_opt = {h: [self.a[(h,t)].getAttr("x") \
                              for t in range(self.T)] for h in self.res}
            return

#%% Distributed Solution Approach

class Home:
    def __init__(self,cost,homedata,p_est,p_sch,gamma):
        self.c = cost
        self.T = len(cost)
        self.data = homedata
        
        self.model = grb.Model(name="Get Optimal Schedule")
        self.model.ModelSense = grb.GRB.MINIMIZE
        self.add_SL()
        self.add_fixed()
        self.netload_var()
        self.set_objective(p_est,p_sch,gamma)
        return
    
    def variables(self):
        self.g = {t:self.model.addVar(vtype=grb.GRB.BINARY,name="g_{0}".format(t)) \
                  for t in range(self.T)}
        
        # Add constraints for the variables
        self.model.addConstr(grb.quicksum([self.g[t] for t in range(self.T)]) == 1)
        return
    
    def netload_var(self):
        self.g = {t:self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                      name="g_{0}".format(t)) \
                  for t in range(self.T)}
        
        # Add constraints for the variables
        devices = [d for dtype in self.data if dtype != "ORG" \
                    for d in self.data[dtype]]
        for t in range(self.T):
            self.model.addConstr(
                self.g[t] == grb.quicksum([self.p[(d,t)] for d in devices]))
        return
    
    def add_SL(self):
        u = {}
        self.p = {}
        SL_data = self.data["SL"]
        for d in SL_data:
            # Schedulable Load parameters
            prate = SL_data[d]["rating"]
            tschd = SL_data[d]["time"]
            for t in range(self.T):
                # Variables
                u[(d,t)] = self.model.addVar(vtype=grb.GRB.BINARY,
                                name="u_{0}_{1}".format(d,t))
                self.p[(d,t)] = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                name="p_{0}_{1}".format(d,t))
                
            # Constraints
            self.model.addConstr(
                grb.quicksum([u[(d,t)] for t in range(self.T)]) == tschd)
            for t in range(self.T):
                self.model.addConstr(self.p[(d,t)] == prate*u[(d,t)])
            for k in range(self.T-tschd):
                if k==0:
                    self.model.addConstr(
                        grb.quicksum([u[(d,t)] \
                                      for t in range(k,k+tschd)]) >= \
                            tschd*u[(d,k)])
                else:
                    self.model.addConstr(
                        grb.quicksum([u[(d,t)] \
                                      for t in range(k,k+tschd)]) >= \
                            tschd*(u[(d,k)]-u[(d,k-1)]))
        return
    
    def add_fixed(self):
        fixed_data = self.data["FIXED"]
        for d in fixed_data:
            for t in range(self.T):
                self.p[(d,t)] = fixed_data[d][t]
        return
    
    def set_objective(self,p_util,p_res,gamma):
        # main objective
        obj1 = grb.quicksum([(self.c[t]*self.g[t]) for t in range(self.T)])
        
        # admm penalty
        kappa = 5
        obj2 = (kappa/2.0) * grb.quicksum([self.g[t] * self.g[t] for t in range(self.T)])
        a = [gamma[t] + (kappa/2.0)*(p_util[t] + p_res[t])\
             for t in range(self.T)]
        obj3 = (kappa/2.0) * grb.quicksum([self.g[t] * a[t] for t in range(self.T)])
        
        # total objective
        self.model.setObjective(obj1+obj2-obj3)
        return
    
    def solve(self,grbpath):
        # Write the LP problem
        self.model.write(grbpath+"load-schedule-agent.lp")
        
        # Set up solver settings
        grb.setParam('OutputFlag', 0)
        grb.setParam('Heuristics', 0)
        
        # Open log file
        logfile = open(grbpath+'gurobi-agent.log', 'w')
        
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
            fixed = ['base', 'hvac', 'hoth2o']
            devices = [d for dtype in self.data if dtype != "ORG" \
                        for d in self.data[dtype] if d not in fixed]
            p_opt = {(d,t): self.p[(d,t)].getAttr("x") for d in devices \
                      for t in range(self.T)}
        
            # Store optimal schedule in the attribute
            self.p_sch = {d: [p_opt[(d,t)] for t in range(self.T)] for d in devices}
            self.g_opt = [self.g[t].getAttr("x") for t in range(self.T)]
            return
        

class Utility:
    def __init__(self,graph,P_util,P_sch,Gamma):
        self.nodes = [n for n in graph if graph.nodes[n]['label'] != 'S']
        self.res = [n for n in graph if graph.nodes[n]['label'] == 'H']
        self.N = len(self.nodes)
        self.T = len(Gamma[self.res[0]])
        
        self.model = grb.Model(name="Get Optimal Utility Estimated Schedule")
        self.model.ModelSense = grb.GRB.MINIMIZE
        self.variables()
        self.network(graph)
        self.set_objective(P_util,P_sch,Gamma)
        return
    
    def variables(self):
        self.g = {(n,t):self.model.addVar(vtype=grb.GRB.BINARY,
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
    
    def set_objective(self,p_util,p_res,gamma):
        # admm penalty
        kappa = 5
        obj1 = (kappa/2.0) * grb.quicksum([self.g[(n,t)] * self.g[(n,t)] \
                                           for n in self.res for t in range(self.T)])
        obj2 = 0
        for n in self.res:
            a = [gamma[n][t] - (kappa/2.0)*(p_util[n][t] + p_res[n][t])\
                 for t in range(self.T)]
            obj2 += (kappa/2.0) * grb.quicksum([self.g[(n,t)] * a[t] \
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














