# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 16:02:13 2022

Author: Rounak Meyur

Description: 
    Tests to check whether the optimization programs are running properly
"""

from revs_fixture import REVS


fx = REVS('runTest')
fx.netID = 121144
fx.com = 2
fx.optim = "individual"

fx.out_dir = f"./out/{fx.netID}-com{fx.com}/{fx.optim}"
fx.fig_dir = "./figs"
fx.grb_dir = f"./gurobi/{fx.netID}-com{fx.com}/{fx.optim}"



tariff, homes, dist, param_data = fx.read_inputs(
    networkID = 121144, 
    save_data=True,
    rating = 3600,
    )
fx.assertIsNotNone(tariff)
fx.assertIsNotNone(homes)
fx.assertIsNotNone(dist)
fx.assertIsNotNone(param_data)

if fx.optim == "individual":
    p_schedule, ev_schedule, soc_schedule = fx.get_individual_optimal(
        tariff, homes, **param_data)
elif fx.optim == "centralized":
    p_schedule, ev_schedule, soc_schedule = fx.get_centralized_optimal(
        tariff, homes, dist, **param_data)
elif fx.optim == "distributed":
    p_schedule, ev_schedule, soc_schedule = fx.get_distributed_optimal(
        tariff, homes, dist, **param_data)

fx.assertIsNotNone(p_schedule)
fx.assertIsNotNone(ev_schedule)
fx.assertIsNotNone(soc_schedule)

fx.plot_result(p_schedule, dist, **param_data, 
               figsize=(60,40), 
               suptitle_sfx = "line loading and node voltages",
               file_name_sfx = "flow_volt",
               fontsize = 70, labelsize = 60, tick_labelsize = 40)