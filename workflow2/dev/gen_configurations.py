#!/usr/bin/env python3

import ast
import configparser
import yaml

import argparse

from swellex import environment as swellex_env



# LOW-LEVEL CONFIGS ============================================================
SEED = 2009
N_RESTARTS = 20
N_SAMPLES = 512

# 1. SEARCH SPACE ==============================================================
parameters = [
    {
        "name": "rec_r",
        "type": "range",
        "bounds": [0.1, 10.0],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "src_z",
        "type": "range",
        "bounds": [1.0, 200.0],
        "value_type": "float",
        "log_scale": False,
    }
]

# 2. OBJECTIVE FUNCTION ========================================================
obj_func_parameters = {
    # "K": K, # Covariance matrix of measured data
    "env_parameters": swellex_env
}

# 3. SEARCH STRATEGY ===========================================================
num_trials = [3, 3]
strategy = "grid"





config = configparser.ConfigParser()
# config["SEARCH_SPACE"] = {}
# config["SEARCH_SPACE"]["SPACE_0"] = parameters[0]

# for i, param in enumerate(parameters):
#     config[f"SEARCH_PARAM_{i}"] = param

# with open("./Source/workflow2/search_space.ini", "w") as f:
#     config.write(f)

config.read("./Source/workflow2/config.ini")





# params = [{k: v for k, v in config[sect].items()} for sect in config.sections()]







# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description="Specify which experiments/simulations to configure."
#     )
#     parser.add_argument(
#         "mode",
#         type=str,
#         help="Specify optimization problem (r = range est., l = localization).",
#         choices=["r", "l"]
#     )





def get_parameters(config):
    parameters = []
    for sect in [s for s in config.sections() if "SEARCH_PARAM" in s]:
        parameters.append(
            {
                "name": config[sect].get("name"),
                "type": config[sect].get("type"),
                "bounds": ast.literal_eval(config[sect].get("bounds")),
                "value_type": config[sect].get("float"),
                "log_scale": config[sect].getboolean("log_scale")
            }
        )
    return parameters

params = get_parameters(config)
print(params)