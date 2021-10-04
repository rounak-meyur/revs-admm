# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 16:15:56 2021

Author: Swapna Thorve and Rounak Meyur

Description: Residence Load Device models
"""

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
    def __init__(self):
        return
    
    def initialize(self,w):
        self.model = grb.Model(name="Get Optimal Schedule")
        self.model.ModelSense = grb.GRB.MINIMIZE
        self.P = {}
        self.w = w
        return

    def add_PV(self,data,solar):
        """
        Add the PV generators as negative loads for the residences. For every 
        customer, if there is one or more PV generators, a power usage variable
        is added for the time window. The upper bound is 0.0 and lower bound is
        the capacity of the PV scaled to the solar irradiance data for the time
        interval.

        Parameters
        ----------
        data : dictionary
            dictionary of residence load device data for different residences.
            key: home ID
            value: dictionary of different load devices
                keys: device category
                values: device settings, parameters
        
        solar: list of float data
            solar irradiance data for the given time window in fraction of 
            maximum value, i.e., 1 unit of solar irradiance implies that the PV
            generate generates at full capacity.

        Returns
        -------
        None.

        """
        for i in data:
            PV_data = data[i]["PV"]
            for d in range(len(PV_data)):
                # PV generator parameters
                PVcap = PV_data[d]
                for t in range(self.w):
                    self.P[(i,d,t)] = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                        ub = 0.0, lb = -PVcap * solar[t],
                                        name="P_{0}_{1}^{2}".format(i,d,t))
        return
    
    def add_ESS(self,data):
        for i in data:
            ESS_data = data[i]["ESS"]
            for d in range(len(ESS_data)):
                # ESS parameters
                modes = ESS_data[d]["charging_modes"]
                pcap = ESS_data[d]["rating"]
                qcap = ESS_data[d]["capacity"]
                soc_init = ESS_data[d]["initial_charge"]*qcap
                
                # Variables
                for t in range(self.w):
                    self.P[(i,d,t)] = pcap * self.model.addVar(vtype=grb.GRB.INTEGER,
                                        ub = modes, lb = -modes,
                                        name="P_{0}_{1}^{2}".format(i,d,t))
                    self.x[(i,d,t)] = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                        lb = 0.2*qcap, ub = 0.8*qcap,
                                        name="x_{0}_{1}^{2}".format(i,d,t))
                    
                # Constraints
                for t in range(self.w):
                    if t == 0:
                        self.model.addConstr(self.x[(i,d,t)] == soc_init)
                    elif t == self.w-1:
                        self.model.addConstr(self.x[(i,d,t)] == soc_init)
                    else:
                        self.model.addConstr(
                            self.x[(i,d,t)] == self.x[(i,d,t-1)] + self.P[(i,d,t)])
        return
    
    def add_SL(self,data):
        u = {}
        for i in data:
            SL_data = data[i]["SL"]
            for d in range(len(SL_data)):
                # Schedulable Load parameters
                prate = SL_data[d]["rating"]
                tschd = SL_data[d]["num_interval"]
                st_init = SL_data[d]["initial_status"]
                for t in range(self.w):
                    # Variables
                    u[(i,d,t)] = self.model.addVar(vtype=grb.GRB.BINARY,
                                    name="u_{0}_{1}^{2}".format(i,d,t))
                    self.P[(i,d,t)] = self.model.addVar(vtype=grb.GRB.CONTINUOUS,
                                    name="P_{0}_{1}^{2}".format(i,d,t))
                    
                # Constraints
                self.model.addConstr(
                    grb.quicksum([u[(i,d,t)] for t in range(self.w)]) == tschd)
                for t in range()
                    
        return
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    