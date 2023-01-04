# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 16:02:13 2022

Author: Rounak Meyur

Description: 
    Tests to check whether the optimization programs are running properly
"""

import sys
import yaml
from revs_fixture import REVS

config_file = sys.argv[1]
with open(config_file) as f:
    config = yaml.safe_load(f)




file_params = config["run_parameters"]["input_filepath"]
inp_params = config["run_parameters"]["input_parameters"]
opt_params = config["run_parameters"]["optimizer_parameters"]
draw_params = config["run_parameters"]["draw_parameters"]



# Initialize file and directory paths
fx = REVS(**file_params)

# Get the inputs for the pipeline
tariff, homes, dist, save_data = fx.read_inputs(**inp_params)


# Perform required optimization
opt_params.update(save_data)
opt_params.update(inp_params)

if fx.optim == "individual":
    p_schedule, ev_schedule, soc_schedule = fx.get_individual_optimal(
        tariff, homes, save = True, **opt_params)
elif fx.optim == "centralized":
    p_schedule, ev_schedule, soc_schedule = fx.get_centralized_optimal(
        tariff, homes, dist, **opt_params)
elif fx.optim == "distributed":
    p_schedule, ev_schedule, soc_schedule = fx.get_distributed_optimal(
        tariff, homes, dist, **opt_params)



rating = inp_params["rating"]
adoption = inp_params["adoption"]

draw_params.update(save_data)
draw_params.update(inp_params)

fx.plot_result(p_schedule, dist, **draw_params, 
               figsize=(60,40), 
               suptitle_sfx = f"rating={rating}Watts : adoption={adoption}%",
               file_name_sfx = f"rate{rating}_adopt{adoption}_flow_volt")