# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 16:15:56 2021

Author: Swapna Thorve and Rounak Meyur

Description: Residence Load Device models
"""

import sys
import gurobipy as grb

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

class Load:
    def __init__(self,w,homedata):
        self.model = grb.Model(name="Get Optimal Schedule")
        self.model.ModelSense = grb.GRB.MINIMIZE
        self.p = {}
        self.g = {}
        self.data = homedata
        self.w = w
        
        # Add all load devices
        self.add_PV()
        self.add_ESS()
        self.add_SL()
        self.add_fixed()
        self.grid_power()
        
        return

    def add_PV(self):
        """
        Add the PV generators as negative loads for the residences. For every 
        customer, if there is one or more PV generators, a power usage variable
        is added for the time window. The upper bound is 0.0 and lower bound is
        the capacity of the PV scaled to the solar irradiance data for the time
        interval.
        """
        PV_data = self.data["PV"]
        for d in PV_data:
            # PV generator parameters
            PVcap = PV_data[d]["rating"]
            ghi = PV_data[d]["solar"]
            for t in range(self.w):
                self.p[(d,t)] = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                    ub = 0.0, lb = -min(PVcap,ghi[t]/1000.0),
                                    name="p_{0}_{1}".format(d,t))
        return
    
    def add_ESS(self):
        x = {}
        c = {}
        ESS_data = self.data["ESS"]
        for d in ESS_data:
            # ESS parameters
            modes = ESS_data[d]["modes"]
            pcap = ESS_data[d]["rating"]
            qcap = ESS_data[d]["capacity"]
            soc_init = ESS_data[d]["initial"]*qcap
            
            # Variables
            for t in range(self.w):
                c[(d,t)] = self.model.addVar(vtype=grb.GRB.INTEGER,
                                    ub = modes, lb = -modes,
                                    name="c_{0}_{1}".format(d,t))
                self.p[(d,t)] = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                    ub = modes*pcap, lb = -modes*pcap,
                                    name="p_{0}_{1}".format(d,t))
                x[(d,t)] = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                lb = 0.1*qcap, ub = 0.9*qcap,
                                name="x_{0}_{1}".format(d,t))
                
            # Constraints
            for t in range(self.w):
                self.model.addConstr(self.p[(d,t)] == (pcap/modes)*c[(d,t)])
                if t==0:
                    self.model.addConstr(x[(d,t)] == x[(d,self.w-1)] + self.p[(d,t)])
                else:
                    self.model.addConstr(x[(d,t)] == x[(d,t-1)] + self.p[(d,t)])
            self.model.addConstr(x[(d,self.w-1)] == soc_init)
        return
    
    def add_SL(self):
        u = {}
        SL_data = self.data["SL"]
        for d in SL_data:
            # Schedulable Load parameters
            prate = SL_data[d]["rating"]
            tschd = SL_data[d]["time"]
            for t in range(self.w):
                # Variables
                u[(d,t)] = self.model.addVar(vtype=grb.GRB.BINARY,
                                name="u_{0}_{1}".format(d,t))
                self.p[(d,t)] = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                name="p_{0}_{1}".format(d,t))
                
            # Constraints
            self.model.addConstr(
                grb.quicksum([u[(d,t)] for t in range(self.w)]) == tschd)
            for t in range(self.w):
                self.model.addConstr(self.p[(d,t)] == prate*u[(d,t)])
            for k in range(self.w-tschd):
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
            for t in range(self.w):
                self.p[(d,t)] = fixed_data[d][t]
        return
    
    def grid_power(self):
        self.z1 = {}
        self.z2 = {}
        devices = [d for dtype in self.data for d in self.data[dtype]]
        for t in range(self.w):
            self.g[t] = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                name="g_{0}".format(t))
            self.z1[t] = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                name="z1_{0}".format(t))
            self.z2[t] = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                name="z2_{0}".format(t))
            self.model.addConstr(
                self.g[t] == grb.quicksum([self.p[(d,t)] for d in devices]))
            self.model.addConstr(self.z1[t] == grb.max_(self.g[t],0))
            self.model.addConstr(self.z2[t] == grb.min_(self.g[t],0))
        return
    
    def set_objective(self,cost,feed,alpha):
        obj = grb.quicksum([(cost[t]-alpha[t])*self.z1[t] for t in range(self.w)]) \
        + grb.quicksum([feed*self.z2[t] for t in range(self.w)])
        self.model.setObjective(obj)
        return
    
    def solve(self,h,grbpath):
        # Write the LP problem
        self.model.write(grbpath+"load-device-"+str(h)+".lp")
        
        # Set up solver settings
        grb.setParam('OutputFlag', 0)
        grb.setParam('Heuristics', 0)
        
        # Open log file
        logfile = open(grbpath+'gurobi-'+str(h)+'.log', 'w')
        
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
            devices = [d for dtype in self.data for d in self.data[dtype] if d not in fixed]
            p_opt = {(d,t): self.p[(d,t)].getAttr("x") for d in devices for t in range(self.w)}
            
            # Store optimal schedule in the attribute
            self.p_sch = {d:[p_opt[(d,t)] for t in range(self.w)] for d in devices}
            return [sum([p_opt[(d,t)] for d in devices]) + sum([self.p[(d,t)] for d in fixed])\
                    for t in range(self.w)]

