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
from tqdm import tqdm

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
            for t in range(self.T+1):
                self.s[t] = self.model.addVar(vtype=grb.GRB.CONTINUOUS, lb=init, 
                                              ub=1.0,name="s_{0}".format(t))
            
                if t == 0:
                    self.model.addConstr(self.s[t] == init)
                else:
                    self.model.addConstr(self.s[t] == self.s[t-1] + (self.p[t-1]/qcap))
            self.model.addConstr(self.s[self.T] >= 0.9)
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
        self.model.write(f"{grbpath}/ev-schedule-agent.lp")
        
        # Set up solver settings
        grb.setParam('OutputFlag', 0)
        grb.setParam('Heuristics', 0)
        
        # Open log file
        logfile = open(f"{grbpath}/gurobi-ev-agent.log", 'w')
        
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
        self.model.write(f"{grbpath}/load-schedule-utility.lp")
        
        # Set up solver settings
        grb.setParam('OutputFlag', 0)
        grb.setParam('Heuristics', 0)
        
        # Open log file
        logfile = open(f"{grbpath}/gurobi-utility.log", 'w')
        
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


#%% Iterative ADMM for distributed optimization
def solve_ADMM(homes,graph,cost,grbpath,
               kappa=5.0,iter_max=15,vset=1.0,vlow=0.95,vhigh=1.05):
    P_est = {0:{h:np.array([0]*len(cost)) for h in homes}}
    P_sch = {0:{h:np.array([0]*len(cost)) for h in homes}}
    G = {0:{h:np.array([0]*len(cost)) for h in homes}}
    S = {}
    C = {}
    
    diff = {}
    
    # ADMM iterations 
    k = 0 # Iteration count
    while(k < iter_max):
        # solve utility level problem to get estimate
        U_obj = Utility(graph,P_est[k],P_sch[k],G[k],
                        kappa=kappa,vset=vset,low=vlow,high=vhigh)
        U_obj.solve(grbpath)
        P_est[k+1] = U_obj.g_opt
        
        
        # solve individual agent level problem
        homelist = [h for h in homes]
        P_sch[k+1] = {}
        S[k+1] = {}
        C[k+1] = {}
        G[k+1] = {}
        diff[k+1] = {}
        for i in tqdm(range(len(homelist)), 
                      desc=f"Iteration {k+1}: Solving home schedule"):
            # solve each individual residence
            h = homelist[i]
            H_obj = Home(cost,homes[h],P_est[k][h],P_sch[k][h],G[k][h],kappa=kappa)
            H_obj.solve(grbpath)
            P_sch[k+1][h] = H_obj.g_opt
            S[k+1][h] = H_obj.p_opt
            C[k+1][h] = H_obj.s_opt
            
            # update dual variables
            check = [(P_est[k+1][h][t] - P_sch[k+1][h][t]) \
                     for t in range(len(cost))]
            G[k+1][h] = [G[k][h][t] + (kappa/2) * check[t] \
                         for t in range(len(cost))]
            diff[k+1][h] = np.linalg.norm(np.array(check))/len(cost)
        
        
        k = k + 1 # Increment iteration
    
    # Return results 
    return diff, P_sch[k],S[k],C[k]


#%% Individual Residence and Centralized Problems

def setup_solve(model, path, location):
    # Write the LP problem
    model.write(f"{path}/ev-schedule-{location}.lp")
    
    # Set up solver settings
    grb.setParam('OutputFlag', 0)
    grb.setParam('Heuristics', 0)
    
    # Open log file
    logfile = open(f"{path}/gurobi-ev-{location}.log", 'w')
    
    # Pass data into my callback function
    model._lastiter = -grb.GRB.INFINITY
    model._lastnode = -grb.GRB.INFINITY
    model._logfile = logfile
    model._vars = model.getVars()
    
    # Solve model and capture solution information
    model.optimize(mycallback)
    
    # Close log file
    logfile.close()
    return

def add_home_noEV(model, T, home_index = 0):
    
    # EV charging demand for T periods
    ev_demand = model.addMVar(T,
        vtype = grb.GRB.CONTINUOUS, 
        name = f"p{home_index}")
    
    # State of charge for T+1 instants
    soc = model.addMVar(
        T+1, 
        vtype = grb.GRB.CONTINUOUS, 
        name = f"s{home_index}")
    
    # Constraints
    model.addConstr(ev_demand == 0)
    model.addConstr(soc == 0)
    
    return ev_demand, soc

def add_home_EV(model, T, home_index = 0, **data):
    
    # get keyword arguments
    prate = data.get("rating")
    qcap = data.get("capacity")
    init = data.get("initial")
    start = data.get("start")
    end = data.get("end")
    
    # EV charging status for T periods
    ev_status = model.addMVar(T,
        vtype = grb.GRB.BINARY, 
        name = f"e{home_index}")
    
    # EV charging demand for T periods
    ev_demand = model.addMVar(T,
        vtype = grb.GRB.CONTINUOUS, 
        name = f"p{home_index}")
    
    # State of charge for T+1 instants
    soc = model.addMVar(
        T+1, 
        vtype = grb.GRB.CONTINUOUS, 
        lb = init, 
        ub = 1.0,
        name = f"s{home_index}")
    
    # Charging constraints
    model.addConstr( ev_demand == ev_status * prate )
    
    for t in range(T):
        if (t < start) or (t >= end):
            model.addConstr( ev_status[t] == 0 )
    
    # soc evolution constraint
    for t in range(T+1):
        if t == 0:
            model.addConstr(soc[t] == init)
        else:
            model.addConstr( soc[t] == soc[t-1] + (ev_demand[t-1] / qcap) )
    
    return ev_demand, soc, ev_status

def add_home_load(model, T, demand, load, ev_demand):
    # add constraint for total demand
    model.addConstr( demand == ev_demand + np.array(load) )
    return

def network_constraints(model, dist, T, vset, vmin, vmax):
    nodes = [n for n in dist.nodes if dist.nodes[n]['label'] != 'S']
    res = [n for n in dist if dist.nodes[n]['label'] == 'H']
    num_res = len(res)
    
    R = compute_Rmat(dist)
    vlow = (vmin*vmin - vset*vset) * np.ones(shape=(num_res,T))
    vhigh = (vmax*vmax - vset*vset) * np.ones(shape=(num_res,T))
    resind = [nodes.index(n) for n in res]
    R_res = R[resind,:][:,resind]
    
    demand = model.addMVar(
        shape=(num_res,T), 
        name = "g", 
        vtype=grb.GRB.CONTINUOUS
        )
    for t in range(T):
        model.addConstr( -R_res @ demand[:,t] <= vhigh[:,t] )
        model.addConstr( -R_res @ demand[:,t] >= vlow[:,t] )
    return demand

def objective_individual_home(model, tariff, demand, soc, T):
    # cost of consumption
    obj_cost = np.array(tariff) @ demand
    
    # soc as close to full charge as possible
    obj_charge = 1 - soc[T]
    
    # total objective
    model.setObjective( (0.01 * obj_cost) + (0.99 * obj_charge) )
    return

def objective_centralized(model, tariff, demand):
    N, T = demand.shape
    a = np.ones(shape=(N,))
    b = np.array(tariff)
    version = grb.gurobi.version()
    if version[0] == 10:
        model.setObjective(a @ demand @ b)
    else:
        model.setObjective(
            sum(b[j] * a @ demand[:, j] for j in range(T)))
    return

def solve_residence(tariff, data, path):
    # initialize model
    model = grb.Model(name = "Get Individual Optimal Schedule")
    model.ModelSense = grb.GRB.MINIMIZE
    
    # size of variables
    T = len(tariff)
    
    # add variables and constraints
    if data["EV"] == {}:
        p, s = add_home_noEV(model, T)  
    else:
        p, s, e = add_home_EV(model, T, **data["EV"])
    
    g = model.addMVar(T, vtype = grb.GRB.CONTINUOUS, name = "g")
    add_home_load(model, T, g, data["LOAD"], p)
    
    # objective function
    objective_individual_home(model, tariff, g, s, T)
    
    # Solve the problem
    setup_solve(model, path, "agent")
    
    if model.SolCount == 0:
        print(f"No solution found, optimization status = {model.Status}")
        sys.exit(0)
    else:
        p_opt = p.getAttr("x")
        s_opt = s.getAttr("x")
        g_opt = g.getAttr("x")
        return p_opt, s_opt, g_opt
    
    
def solve_central(tariff, homes, dist, path, vset, vmin, vmax):
    # initialize model
    model = grb.Model(name = "Get Central Optimal Schedule")
    model.ModelSense = grb.GRB.MINIMIZE
    
    # size of variables
    T = len(tariff)
    res = [n for n in dist if dist.nodes[n]['label'] == 'H']
    
    # network constraints
    g = network_constraints(model, dist, T, vset, vmin, vmax)
    
    # add residence variables and constraints
    p = {}
    s = {}
    e = {}
    for i,h in enumerate(res):
        if homes[h]["EV"] == {}:
            p[h], s[h] = add_home_noEV(
                model, T, home_index = i)  
        else:
            p[h], s[h], e[h] = add_home_EV(
                model, T, home_index = i, **homes[h]["EV"])
    
        add_home_load(model, T, g[i,:], homes[h]["LOAD"], p[h])
    
    # objective function
    objective_centralized(model, tariff, g)
    
    # Solve the problem
    setup_solve(model, path, "central")
    
    if model.SolCount == 0:
        print(f"No solution found, optimization status = {model.Status}")
        sys.exit(0)
    else:
        p_opt = {h: p[h].getAttr("x") for h in res}
        s_opt = {h: s[h].getAttr("x") for h in res}
        g_opt = {h: g.getAttr("x")[i,:] for i,h in enumerate(res)}
        return p_opt, s_opt, g_opt
