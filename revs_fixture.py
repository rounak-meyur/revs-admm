# -*- coding: utf-8 -*-
"""
Created on Fri Feb  4 12:37:23 2022

Author: Rounak Meyur
Description: This compuutes the optimal schedule when each residence solves its
individual optimization problem in a selfish manner.
"""

import logging
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")



import os
import unittest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm

from extract import GetTariff, GetHomeLoad, GetDistNet, GetCommunity, get_homes_ev_param
from extract import combine_result
from lpsolver import solve_residence, solve_ADMM, solve_central
from drawing import boxplot_flow, boxplot_volt



def get_fig_from_ax(ax, **kwargs):
    if not ax:
        no_ax = True
        ndim = kwargs.get('ndim', (1, 1))
        figsize = kwargs.get('figsize', (10, 10))
        constrained_layout = kwargs.get('constrained_layout', False)
        fig, ax = plt.subplots(*ndim, figsize=figsize, 
                               constrained_layout=constrained_layout)
    else:
        no_ax = False
        if not isinstance(ax, matplotlib.axes.Axes):
            if isinstance(ax, list):
                getter = kwargs.get('ax_getter', lambda x: x[0])
                ax = getter(ax)
            if isinstance(ax, dict):
                getter = kwargs.get('ax_getter', lambda x: next(iter(ax.values())))
                ax = getter(ax)
        fig = ax.get_figure()

    return fig, ax, no_ax


def close_fig(fig, to_file=None, show=True, **kwargs):
    if to_file:
        fig.savefig(to_file, **kwargs)
    if show:
        plt.show()
    plt.close(fig)
    pass


class REVS(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        
        self.netID = 121144
        self.regID = 121
        self.com = 2
        self.optim = "individual"
        
        self.data_path = "./input"
        self.out_dir = "./out"
        self.fig_dir = "./figs"
        self.grb_dir = "./gurobi"
        
        pass
    
    # Out directory setter/ if not, create a directory
    @property
    def out_dir(self):
        return self._out_dir

    @out_dir.setter
    def out_dir(self, out):
        self._out_dir = out
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        pass
    
    # Figures directory setter/ if not, create a directory
    @property
    def fig_dir(self):
        return self._fig_dir

    @fig_dir.setter
    def fig_dir(self, fig):
        self._fig_dir = fig
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)
        pass
    
    # Gurobi directory setter/ if not, create a directory
    @property
    def grb_dir(self):
        return self._grb_dir

    @grb_dir.setter
    def grb_dir(self, grb):
        self._grb_dir = grb
        if not os.path.exists(self.grb_dir):
            os.makedirs(self.grb_dir)
        pass
    
    
    # functions to read input data
    def read_tariff(self, tariffID = None, shift = 6):
        if not tariffID:
            tariffID = "DVP"
            
        tariff = GetTariff(self.data_path, tariffID, shift)
        return tariff
    
    def read_homes(self, regionID = None, shift = 6):
        if not regionID:
            regionID = self.regID
        
        # Home load data
        homes = GetHomeLoad(self.data_path, regionID, shift=shift)
        return homes
    
    def read_network(self, networkID = None):
        if not networkID:
            networkID = self.netID
        
        # Distribution network data
        dist = GetDistNet(self.data_path, networkID)
        return dist
    
    def read_community(self, networkID = None, com_index = 2):
        if not networkID:
            networkID = self.netID
        
        # Get the community of residences
        filename = f"{self.data_path}/{networkID}-com.txt"
        com = GetCommunity(filename, com_index)
        return com
    
    def read_inputs(
            self, regionID = None, networkID = None, tariffID = None,
            ev_homes = None, **kwargs):
        # Get keyword arguments
        adoption = kwargs.get("adoption", 90) * 1e-2
        rating = kwargs.get("rating", 4.8)
        capacity = kwargs.get("capacity", 20)
        initial = kwargs.get("initial_soc", 0.2)
        start = kwargs.get("start_time", 11)
        end = kwargs.get("end_time", 23)
        sh = kwargs.get("shift_time", 6)
        seed = kwargs.get("seed", 1234)
        vset = kwargs.get("v0", 1.03)
        
        # Get the electricity tariff
        tariff = self.read_tariff(tariffID = tariffID, shift = sh)
        
        # Get the homes and hourly energy consumption
        all_homes = self.read_homes(regionID = regionID, shift = sh)
        
        # Get the power distribution network
        dist = self.read_network(networkID = networkID)
        
        # Get EV adopting homes if not present
        com = self.read_community(
            networkID = networkID, 
            com_index = self.com
            )
        if not ev_homes:
            np.random.seed(int(seed))
            num_choice = int(adoption * len(com))
            ev_homes = np.random.choice(com, num_choice, replace=False)
        
        # Get the home data with EV charging parameters
        homes = get_homes_ev_param(
            all_homes, dist, ev_homes,
            rating, capacity, initial, start, end)
        
        # save data for writing in output file
        save = kwargs.get("save_data", False)
        if save:
            save_home_data = dict(
                save_data = True,
                adoption = int(adoption * 100),
                rating = int(rating * 1000),
                ev_homes = ev_homes,
                community = com,
                seed = seed,
                shift = sh,
                start = start,
                end = end,
                initial = initial,
                vset = vset,
                )
            return tariff, homes, dist, save_home_data
        
        return tariff, homes, dist
    
    
    def get_individual_optimal(self, tariff, homes, **kwargs):
        # keyword arguments
        save = kwargs.get("save_data", False)
        ev_homes = kwargs.get("ev_homes", None)
        adopt = kwargs.get("adoption", 90)
        rating = kwargs.get("rating", 4800)
        seed = kwargs.get("seed", None)
        
        # Solve the problem
        Pres = {}
        Pev = {}
        soc = {}
        homelist = list(homes.keys())
        for i in tqdm(range(len(homelist)), desc="Solving home schedule"):
            h = homelist[i]
            p_opt, s_opt, g_opt = solve_residence(
                tariff, homes[h], 
                f"{self.grb_dir}")
            Pres[h] = g_opt
            Pev[h] = p_opt
            soc[h] = s_opt
        
        # save the results
        if save:
            data = combine_result(Pres, Pev, soc, ev_homes)
            filename = f"adopt{adopt}-rating{rating}-seed{seed}.txt"
            with open(f"{self.out_dir}/{filename}", "w") as f:
                f.write(data)
            
        return Pres, Pev, soc
    
    
    def get_centralized_optimal(self, tariff, homes, dist, **kwargs):
        
        vset = kwargs.get("v0", 1.03)
        vmin = kwargs.get("vmin", 0.90)
        vmax = kwargs.get("vmax", 1.05)
        save = kwargs.get("save_data", False)
        ev_homes = kwargs.get("ev_homes", None)
        adopt = kwargs.get("adoption", 90)
        rating = kwargs.get("rating", 4800)
        seed = kwargs.get("seed", None)
        
        # solve the problem
        Pev, soc, Pres = solve_central(
            tariff, homes, dist, f"{self.grb_dir}",
            vset, vmin, vmax)
        
        # save the results
        if save:
            data = combine_result(Pres, Pev, soc, ev_homes)
            filename = f"adopt{adopt}-rating{rating}-seed{seed}.txt"
            with open(f"{self.out_dir}/{filename}", "w") as f:
                f.write(data)
            
        return Pres, Pev, soc
    
    def get_distributed_optimal(self, tariff, homes, dist, **kwargs):
        # get keyword arguments
        kappa = kwargs.get("kappa", 5.0)
        iter_max = kwargs.get("max_iterations", 15)
        vset = kwargs.get("v0", 1.03)
        vlow = kwargs.get("vlow", 0.95)
        vhigh = kwargs.get("vhigh", 1.05)
        
        # get the schedule from distributed optimization
        diff, Pres, Pev, soc = solve_ADMM(
            homes, dist, tariff, 
            f"{self.grb_dir}",
            kappa=kappa, iter_max=iter_max,
            vset=vset, vlow=vlow, vhigh=vhigh
            )
        
        save = kwargs.get("save_data", False)
        ev_homes = kwargs.get("ev_homes", None)
        adopt = kwargs.get("adoption", 90)
        rating = kwargs.get("rating", 4800)
        seed = kwargs.get("seed", None)
        
        if save:
            data = combine_result(Pres, Pev, soc, ev_homes, diff)
            filename = f"adopt{adopt}-rating{rating}-seed{seed}.txt"
            with open(f"{self.out_dir}/{filename}", "w") as f:
                f.write(data)
        
        return Pres, Pev, soc
    
    def plot_result(
            self, demand, dist, 
            ax = None, to_file=None, show=True, 
            plot_list = ["flow", "volt"], 
            **kwargs):
        kwargs.setdefault('figsize', (50, 30))
        fontsize = kwargs.get('fontsize', 30)
        do_return = kwargs.get('do_return', False)
        node_interest = kwargs.get("community", None)

        # ---- PLOT ----
        fig, axs, no_ax = get_fig_from_ax(ax, 
                                          ndim=(len(plot_list),1), **kwargs)
        for i,plot_type in enumerate(plot_list):
            if plot_type == "flow":
                boxplot_flow(demand, dist, axs[i], **kwargs)
            elif plot_type == "volt":
                boxplot_volt(demand, dist, node_interest, axs[i], **kwargs)
        
        # ---- Edit the title of the plot ----
        if file_name_sfx := kwargs.get('file_name_sfx'):
            if not to_file:
                to_file = f"{self.netID}-com{self.com}-{self.optim}-results"
            to_file = f"{to_file}_{file_name_sfx}"

        if no_ax:
            to_file = f"{self.fig_dir}/{to_file}.png"
            suptitle_pfx = f"Region {self.netID} : Community {self.com}"
            suptitle = f"{self.optim} optimization"
            if suptitle_sfx := kwargs.get('suptitle_sfx'):
                suptitle = f"{suptitle_pfx} : {suptitle} : {suptitle_sfx}"

            fig.suptitle(suptitle, fontsize=fontsize)
            close_fig(fig, to_file, show)

        if do_return:
            return fig, ax
        pass
