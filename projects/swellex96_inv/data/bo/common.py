#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path

FREQ = [
    # 49.0,
    # 64.0,
    # 79.0,
    # 94.0,
    # 112.0,
    # 130.0,
    148.0,
    # 166.0,
    # 201.0,
    235.0,
    # 283.0,
    # 338.0,
    388.0,
]
SWELLEX96PATH = Path("../data/swellex96_S5_VLA_inv")

TIME_STEP = 121
# if TIME_STEP == 20:
#     TRUE_R = 5.87
#     TRUE_SRC_Z = 59.0
#     TILT = 0.5
#     H_W = 217.0
# if TIME_STEP == 50:
#     TRUE_R = 4.15
#     TRUE_SRC_Z = 66
#     TILT = 0.5
#     H_W = 217.0
if TIME_STEP == 121:
    TRUE_R = 1.07
    TRUE_SRC_Z = 71.11
    TILT = 2.0
    H_W = 217
TRUE_SIM_VALUES = {
    "rec_r": TRUE_R,
    "src_z": TRUE_SRC_Z,
    "tilt": TILT,
    "h_w": H_W,
    # "z3": 30.0,
    # "dc1": -22.167,
    # "dc2": -7.273,
    # "dc3": -6.0,
    # "dc4": -1.279,
    # "dc5": -1.229,
    "h_sed": 23.0,
    "c_p_sed_top": 1572.3,
    "dc_p_sed": 20.7,
    "c_p_sed_bot": 1593.0,
    "a_p_sed": 0.2,
    "rho_sed": 1.76,
    # "c1": 1522.0,
    # "c2": 1522.0,
    # "c3": 1495.274,
    # "c4": 1490.892,
    # "c5": 1488.394,
    # "c6": 1488.89,
    # "c7": 1488.89,
}
# The following are obtained from running DE with strong priors (narrow bounds).
TRUE_EXP_VALUES = {
    "rec_r": 1.072082,
    "src_z": 71.526224,
    "tilt": 2.067787,
    "h_w": 219.634477,
    # "z3": 30.0,
    # "dc1": -22.167,
    # "dc2": -7.273,
    # "dc3": -6.0,
    # "dc4": -1.279,
    # "dc5": -1.229,
    "h_sed": 20.070524,
    "c_p_sed_top": 1567.331676,
    "dc_p_sed": 22.206878,
    "c_p_sed_bot": 1589.538554,
    "a_p_sed": 0.652053,
    "rho_sed": 1.990237,
    # "c1": 1522.0,
    # "c2": 1522.0,
    # "c3": 1495.274,
    # "c4": 1490.892,
    # "c5": 1488.394,
    # "c6": 1488.89,
    # "c7": 1488.89,
}
# TRUE_VALUES["c2"] = TRUE_VALUES["c1"] + TRUE_VALUES["dc1"]
# TRUE_VALUES["c3"] = TRUE_VALUES["c2"] + TRUE_VALUES["dc2"]
# TRUE_VALUES["c4"] = TRUE_VALUES["c3"] + TRUE_VALUES["dc3"]
# TRUE_VALUES["c5"] = TRUE_VALUES["c4"] + TRUE_VALUES["dc4"]
# TRUE_VALUES["c6"] = TRUE_VALUES["c5"] + TRUE_VALUES["dc5"]

VARIABLES = {
    "rec_r": "$r_\mathrm{src}$ [km]",
    "src_z": "$z_\mathrm{src}$ [m]",
    "h_w": "$h_w$ [m]",
    "tilt": "$\\tau$ [$^\circ$]",
    "h_sed": "$h_s$ [m]",
    "c_p_sed_top": "$c_{s,t}$ [m/s]",
    "dc_p_sed": "$\delta c_s$ [m/s]",
    "c_p_sed_bot": "$c_{s,b}$ [m/s]",
    # "c3": "$c_3$ [m/s]",
    # "c4": "$c_4$ [m/s]",
    # "c5": "$c_5$ [m/s]",
    # "c6": "$c_6$ [m/s]",
    "a_p_sed": "$\\alpha_s$ [dB/$\lambda$]",
    "rho_sed": "$\\rho_s$ [g/cm$^3$]",
    # "dc1": "$\delta c_1$ [m/s]",
    # "dc2": "$\delta c_2$ [m/s]",
    # "dc4": "$\delta c_4$ [m/s]",
    # "dc5": "$\delta c_5$ [m/s]",
    # "c2": "$c_2$ [m/s]",
    # "dc3": "$\delta c_3$ [m/s]",
    # "z3": "$z_3$ [m]",
}

SEARCH_SPACE = [
        {"name": "rec_r", "type": "range", "bounds": [TRUE_R-0.25, TRUE_R + 0.25]},
        {"name": "src_z", "type": "range", "bounds": [60.0, 80.0]},
        {"name": "tilt", "type": "range", "bounds": [-3.0, 3.0]},
        {"name": "h_w", "type": "range", "bounds": [TRUE_SIM_VALUES["h_w"] - 5.0, TRUE_SIM_VALUES["h_w"] + 5.0]},
        {"name": "h_sed", "type": "range", "bounds": [10.0, 40.0]},
        {"name": "c_p_sed_top", "type": "range", "bounds": [1540.0, 1640.0]},
        {"name": "dc_p_sed", "type": "range", "bounds": [0.0, 50.0]},
        {"name": "a_p_sed", "type": "range", "bounds": [0.01, 3.0]},
        {"name": "rho_sed", "type": "range", "bounds": [1.0, 3.0]},
        # {"name": "z3", "type": "range", "bounds": [10.0, 60.0]},
        # {"name": "c3", "type": "range", "bounds": [1480.0, 1520.0]}, # 30 m
        # {"name": "c4", "type": "range", "bounds": [1480.0, 1520.0]}, # 60 m
        # {"name": "c5", "type": "range", "bounds": [1480.0, 1520.0]}, # 100 m
        # {"name": "c6", "type": "range", "bounds": [1480.0, 1520.0]}, # 150 m
        # {"name": "dc3", "type": "range", "bounds": [-20.0, 0.0]},
        # {"name": "c2", "type": "range", "bounds": [1460.0, 1540.0]},
        # {"name": "c5", "type": "range", "bounds": [1460.0, 1500.0]},
        # {"name": "c6", "type": "range", "bounds": [1460.0, 1500.0]},
        # {"name": "dc1", "type": "range", "bounds": [-40.0, 20.0]},
        # {"name": "dc2", "type": "range", "bounds": [-20.0, 10.0]},
        # {"name": "dc3", "type": "range", "bounds": [-10.0, 10.0]},
        # {"name": "dc4", "type": "range", "bounds": [-10.0, 10.0]},
        # {"name": "dc5", "type": "range", "bounds": [-10.0, 10.0]},
        # {"name": "c1", "type": "range", "bounds": [1470.0, 1570.0]},
    ]


STRATEGY_COLORS = {
    "Sobol": "red",
    "UCB": "green",
    "EI": "blue",
    "BAxUS": "orange",
    "Random": "green",
}


@dataclass(frozen=True)
class SWELLEX96Paths:
    path: Path = SWELLEX96PATH
    main_environment_data_sim: Path = SWELLEX96PATH / "env_models" / "main_sim_env.json"
    main_environment_data_exp: Path = SWELLEX96PATH / "env_models" / "main_exp_env.json"
    simple_environment_data: Path = SWELLEX96PATH / "env_models" / "simple_env.json"
    gps_data: Path = SWELLEX96PATH / "gps" / "source_tow.csv"
    acoustic_path: Path = SWELLEX96PATH / "acoustic" / "processed_001"
    outputs: Path = SWELLEX96PATH / "outputs"
    ambiguity_surfaces: Path = SWELLEX96PATH / "acoustic" / "ambiguity_surfaces"
