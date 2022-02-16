# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 10:27:37 2022

Author: Rounak Meyur

Description: Classes and methods to schedule EV charging at residential premises
with and without assuring network reliability
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
        
        if self.data["EV"] == {}:
            # Variables for no EV in residence
            for t in range(self.T):
                self.p[t] = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                              name="p_{0}".format(t))
                self.model.addConstr(self.p[t] == 0)
            for t in range(self.T+1):
                self.s[t] = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                              name="s_{0}".format(t))
                self.model.addConstr(self.s[t] == 0)
        else:
            # Variables for EV in residence
            e = {}
            EV_data = self.data["EV"]
            prate = EV_data['rating']
            qcap = EV_data['capacity']
            init = EV_data['initial']
            start = EV_data['start']
            end = EV_data['end']
            
            # Power consumption variables and constraints
            for t in range(self.T):
                e[t] = self.model.addVar(vtype=grb.GRB.BINARY,
                                             name="e_{0}".format(t))
                self.p[t] = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                               name="p_{0}".format(t))
                self.model.addConstr(self.p[t] == e[t]*prate)
                if (t<start) or (t>=end):
                    self.model.addConstr(e[t] == 0)
            
            # SOC variables and constraints
            self.model.addConstr(self.s[self.T] >= 0.9)
            for t in range(self.T+1):
                self.s[t] = self.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=init, 
                                              ub=1.0,name="s_{0}".format(t))
            
                if t == 0:
                    self.model.addConstr(self.s[t] == init)
                else:
                    self.model.addConstr(self.s[t] == self.s[t-1] + (self.p[t-1]/qcap))
        return
    
    def set_objective(self,p_util,p_res,gamma,kappa=5.0):
        # main objective
        obj1 = grb.quicksum([(self.c[t]*self.g[t]) for t in range(self.T)])
        
        # admm penalty
        obj2 = (kappa/2.0) * grb.quicksum([self.g[t] * self.g[t] for t in range(self.T)])
        a = [gamma[t] + (kappa/2.0)*(p_util[t] + p_res[t])\
             for t in range(self.T)]
        obj3 = grb.quicksum([self.g[t] * a[t] for t in range(self.T)])
        
        # auxillary objective
        # obj_charge = 1.0 - self.s[self.T]
        
        # total objective
        #  obj = 0.001*(obj1+obj2-obj3) + 0.999*obj_charge
        obj = obj1+obj2-obj3
        self.model.setObjective(obj)
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
            self.s_opt = [self.s[t].getAttr("x") for t in range(self.T+1)]
            self.g_opt = [self.g[t].getAttr("x") for t in range(self.T)]
            return
        

class Utility:
    def __init__(self,graph,P_util,P_sch,Gamma,
                 kappa=5.0,vset=1.0,low=0.95,high=1.05):
        self.nodes = [n for n in graph.nodes if graph.nodes[n]['label'] != 'S']
        self.res = [n for n in graph if graph.nodes[n]['label'] == 'H']
        self.N = len(self.nodes)
        self.T = len(Gamma[self.res[0]])
        
        self.model = grb.Model(name="Get Optimal Utility Estimated Schedule")
        self.model.ModelSense = grb.GRB.MINIMIZE
        self.variables()
        self.network(graph,vset=vset,vmin=low,vmax=high)
        self.set_objective(P_util,P_sch,Gamma,kappa=kappa)
        return
    
    def variables(self):
        self.g = self.model.addMVar(shape=(len(self.res),self.T),
                                    name = "g",vtype=grb.GRB.CONTINUOUS)
        return
    
    def network(self,graph,vset=1.0,vmin=0.95,vmax=1.05):
        R = compute_Rmat(graph)
        vlow = (vmin*vmin - vset*vset) * np.ones(shape=(len(self.res),self.T))
        vhigh = (vmax*vmax - vset*vset) * np.ones(shape=(len(self.res),self.T))
        
        resind = [self.nodes.index(n) for n in self.res]
        R_res = R[resind,:][:,resind]
        
        for t in range(self.T):
            self.model.addConstr(R_res@self.g[:,t] <= vhigh[:,t])
            self.model.addConstr(R_res@self.g[:,t] >= vlow[:,t])
        return
    
    def set_objective(self,p_util,p_res,gamma,kappa=5.0):
        # admm penalty
        obj1 = 0
        obj2 = 0
        for i,n in enumerate(self.res):
            obj1 += (kappa/2.0) * (self.g[i,:] @ self.g[i,:])
            a = np.array([gamma[n][t] - (kappa/2.0)*(p_util[n][t] + p_res[n][t])\
                 for t in range(self.T)])
            obj2 += self.g[i,:] @ a
        
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
            G = self.g.getAttr("x").tolist()
            self.g_opt = {h: G[i] for i,h in enumerate(self.res)}
            return


#%% Individual Residence Problem
class Residence:
    def __init__(self,cost,homedata):
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
        self.set_objective()
        return
    
    def netload_var(self):
        for t in range(self.T):
            self.g[t] = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                          name="g_{0}".format(t))
            self.model.addConstr(
                self.g[t] == self.p[t] + self.data["LOAD"][t])
        return
    
    def add_EV(self):
        
        if self.data["EV"] == {}:
            # Variables for no EV in residence
            for t in range(self.T):
                self.p[t] = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                              name="p_{0}".format(t))
                self.model.addConstr(self.p[t] == 0)
            for t in range(self.T+1):
                self.s[t] = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                              name="s_{0}".format(t))
                self.model.addConstr(self.s[t] == 0)
        else:
            # Variables for EV in residence
            e = {}
            EV_data = self.data["EV"]
            prate = EV_data['rating']
            qcap = EV_data['capacity']
            init = EV_data['initial']
            start = EV_data['start']
            end = EV_data['end']
            
            # Power consumption variables and constraints
            for t in range(self.T):
                e[t] = self.model.addVar(vtype=grb.GRB.BINARY,
                                             name="e_{0}".format(t))
                self.p[t] = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                               name="p_{0}".format(t))
                self.model.addConstr(self.p[t] == e[t]*prate)
                if (t<start) or (t>=end):
                    self.model.addConstr(e[t] == 0)
            
            # SOC variables and constraints
            for t in range(self.T+1):
                self.s[t] = self.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=init, 
                                              ub=1.0,name="s_{0}".format(t))
            
                if t == 0:
                    self.model.addConstr(self.s[t] == init)
                else:
                    self.model.addConstr(self.s[t] == self.s[t-1] + (self.p[t-1]/qcap))
        return
    
    def set_objective(self):
        # main objective
        obj1 = grb.quicksum([(self.c[t]*self.g[t]) for t in range(self.T)])
        
        # auxillary objective
        obj_charge = 1.0 - self.s[self.T]
        
        # total objective
        self.model.setObjective(0.01*obj1+0.99*obj_charge)
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
            self.s_opt = [self.s[t].getAttr("x") for t in range(self.T+1)]
            self.g_opt = [self.g[t].getAttr("x") for t in range(self.T)]
            return


#%% Iterative ADMM for distributed optimization
def solve_ADMM(homes,graph,cost,grbpath="",
               kappa=5.0,iter_max=15,vset=1.0,vlow=0.95,vhigh=1.05):
    P_est = {0:{h:[0]*len(cost) for h in homes}}
    P_sch = {0:{h:[0]*len(cost) for h in homes}}
    G = {0:{h:[0]*len(cost) for h in homes}}
    S = {}
    C = {}
    
    diff = {}
    
    # ADMM iterations 
    k = 0 # Iteration count
    while(k <= iter_max):
        # solve utility level problem to get estimate
        U_obj = Utility(graph,P_est[k],P_sch[k],G[k],
                        kappa=kappa,vset=vset,low=vlow,high=vhigh)
        U_obj.solve(grbpath)
        P_est[k+1] = U_obj.g_opt
        
        
        # solve individual agent level problem
        P_sch[k+1] = {}
        S[k+1] = {}
        C[k+1] = {}
        for h in homes:
            H_obj = Home(cost,homes[h],P_est[k][h],P_sch[k][h],G[k][h],kappa=kappa)
            H_obj.solve(grbpath)
            P_sch[k+1][h] = H_obj.g_opt
            S[k+1][h] = H_obj.p_opt
            C[k+1][h] = H_obj.s_opt
            
        
        
        # update dual variables
        G[k+1] = {}
        diff[k+1] = {}
        for h in homes:
            check = [(P_est[k+1][h][t] - P_sch[k+1][h][t]) for t in range(len(cost))]
            G[k+1][h] = [G[k][h][t] + (kappa/2) * check[t] for t in range(len(cost))]
            diff[k+1][h] = np.linalg.norm(np.array(check))/len(cost)
        
        
        k = k + 1 # Increment iteration
        print("Iteration count: ",k)
    
    # Return results 
    return diff, P_sch[k],S[k],C[k]


#%% Centralized MILP solved by operator after access to all EV parameters
class Central:
    def __init__(self,homedata,graph,cost,grbpath="",
                 vset=1.0,vmin=0.95,vmax=1.05):
        self.c = np.array(cost)
        self.T = len(cost)
        self.nodes = [n for n in graph.nodes if graph.nodes[n]['label'] != 'S']
        self.res = [n for n in graph if graph.nodes[n]['label'] == 'H']
        self.N = len(self.nodes)
        self.data = homedata
        
        self.model = grb.Model(name="Get Optimal Schedule MILP")
        self.model.ModelSense = grb.GRB.MINIMIZE
        
        # Add residence constraints
        self.netload_var()
        self.add_EV()
        
        # Add network constraints and incentive
        self.network(graph,vset=vset,vmin=vmin,vmax=vmax)
        self.set_objective()
        
        return
    
    def netload_var(self):
        self.e = self.model.addMVar(shape=(len(self.res),self.T),
                                    name = "e",vtype=grb.GRB.BINARY)
        self.g = self.model.addMVar(shape=(len(self.res),self.T),
                                    name = "g",vtype=grb.GRB.CONTINUOUS)
        self.p = self.model.addMVar(shape=(len(self.res),self.T),
                                    name = "p",vtype=grb.GRB.CONTINUOUS)
        self.s = self.model.addMVar(shape=(len(self.res),self.T+1),ub=1,lb=0,
                                    name = "s",vtype=grb.GRB.CONTINUOUS)
        f = np.array([[self.data[h]["LOAD"][t] for t in range(self.T)] \
                      for h in self.res])
        for t in range(self.T):
            self.model.addConstr(self.g[:,t] == self.p[:,t] + f[:,t])
        return
    
    def add_EV(self):
        for i,h in enumerate(self.res):
            if self.data[h]["EV"] == {}:
                # Variables for no EV in residence
                self.model.addConstr(self.p[i,:] == 0)
                self.model.addConstr(self.e[i,:] == 0)
                self.model.addConstr(self.s[i,:] == 0)
            else:
                # Variables for EV in residence
                EV_data = self.data[h]["EV"]
                prate = EV_data['rating']
                qcap = EV_data['capacity']
                init = EV_data['initial']
                start = EV_data['start']
                end = EV_data['end']
            
                # Power consumption variables and constraints
                for t in range(self.T):
                    self.model.addConstr(self.p[i,t] == self.e[i,t]*prate)
                    if (t<start) or (t>=end):
                        self.model.addConstr(self.e[i,t] == 0)
                
                # SOC variables and constraints
                self.model.addConstr(self.s[i,self.T]>=0.9)
                for t in range(self.T+1):
                    if t == 0:
                        self.model.addConstr(self.s[i,t] == init)
                    else:
                        self.model.addConstr(self.s[i,t] == self.s[i,t-1] \
                                             + (self.p[i,t-1]/qcap))
        return
    
    def network(self,graph,vset=1.0,vmin=0.95,vmax=1.05):
        R = compute_Rmat(graph)
        vlow = (vmin*vmin - vset*vset) * np.ones(shape=(len(self.res),self.T))
        vhigh = (vmax*vmax - vset*vset) * np.ones(shape=(len(self.res),self.T))
        
        resind = [self.nodes.index(n) for n in self.res]
        R_res = R[resind,:][:,resind]
        
        for t in range(self.T):
            self.model.addConstr(-R_res@self.g[:,t] <= vhigh[:,t])
            self.model.addConstr(-R_res@self.g[:,t] >= vlow[:,t])
        return
    
    def set_objective(self):
        obj1 = 0
        for t in range(self.T):
            obj1 += self.g[:,t].sum() * self.c[t]
        
        # Alternate objective function for balancing high EV charging and cost
        # obj2 = len(self.res) - self.s[:,self.T].sum()
        # self.model.setObjective(0.001*obj1+0.999*obj2)
        
        self.model.setObjective(obj1)
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
        
        # Solve model and capture solution information
        self.model.optimize(mycallback)
        
        # Close log file
        logfile.close()
        if self.model.SolCount == 0:
            print('No solution found, optimization status = %d' % self.model.Status)
            sys.exit(0)
        else:
            G = self.g.getAttr("x").tolist()
            P = self.p.getAttr("x").tolist()
            S = self.s.getAttr("x").tolist()
            self.g_opt = {h: G[i] for i,h in enumerate(self.res)}
            self.p_opt = {h: P[i] for i,h in enumerate(self.res)}
            self.s_opt = {h: S[i] for i,h in enumerate(self.res)}
        return
