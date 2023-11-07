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
if TIME_STEP == 20:
    TRUE_R = 5.87
    TRUE_SRC_Z = 59.0
    TILT = 0.5
    H_W = 217.0
if TIME_STEP == 50:
    TRUE_R = 4.15
    TRUE_SRC_Z = 66
    TILT = 0.5
    H_W = 217.0
if TIME_STEP == 121:
    TRUE_R = 1.07
    TRUE_SRC_Z = 71.11
    TILT = 2.0
    H_W = 217.5
TRUE_VALUES = {
    "rec_r": TRUE_R,
    "src_z": TRUE_SRC_Z,
    "tilt": TILT,
    "h_w": H_W,
    # "c1": 1522.0,
    # "dc1": -0.573,
    # "dc2": -21.594,
    # "dc3": -7.273,
    # "dc4": -2.362,
    # "dc5": -1.764,
    "h_sed": 23.0,
    "c_p_sed_top": 1572.3,
    "dc_p_sed": 20.7,
    "c_p_sed_bot": 1593.0,
    "a_p_sed": 0.2,
    "rho_sed": 1.76,
}

VARIABLES = {
    "rec_r": "$r_\mathrm{src}$ [km]",
    "src_z": "$z_\mathrm{src}$ [m]",
    "h_w": "$h_w$ [m]",
    "tilt": "$\\tau$ [$^\circ$]",
    "h_sed": "$h_s$ [m]",
    "c_p_sed_top": "$c_{s,t}$ [m/s]",
    "dc_p_sed": "$c_{s,b}$ [m/s]",
    "a_p_sed": "$\\alpha_s$ [dB/$\lambda$]",
    "rho_sed": "$\\rho_s$ [g/cm$^3$]",
    # "dc1": "$\Delta c_1$ [m/s]",
    # "dc2": "$\Delta c_2$ [m/s]",
    # "dc3": "$\Delta c_3$ [m/s]",
    # "dc4": "$\Delta c_4$ [m/s]",
    # "dc5": "$\Delta c_5$ [m/s]",
}

SEARCH_SPACE = [
        {"name": "rec_r", "type": "range", "bounds": [TRUE_R-0.25, TRUE_R + 0.25]},
        {"name": "src_z", "type": "range", "bounds": [60.0, 80.0]},
        {"name": "tilt", "type": "range", "bounds": [-3.0, 3.0]},
        {"name": "h_w", "type": "range", "bounds": [TRUE_VALUES["h_w"] - 5.0, TRUE_VALUES["h_w"] + 5.0]},
        # {"name": "c1", "type": "range", "bounds": [1470.0, 1570.0]},
        # {"name": "dc1", "type": "range", "bounds": [-10.0, 10.0]},
        # {"name": "dc2", "type": "range", "bounds": [-40.0, 0.0]},
        # {"name": "dc3", "type": "range", "bounds": [-20.0, 0.0]},
        # {"name": "dc4", "type": "range", "bounds": [-10.0, 10.0]},
        # {"name": "dc5", "type": "range", "bounds": [-10.0, 10.0]},
        {"name": "h_sed", "type": "range", "bounds": [10.0, 40.0]},
        {"name": "c_p_sed_top", "type": "range", "bounds": [1540.0, 1640.0]},
        {"name": "dc_p_sed", "type": "range", "bounds": [0.0, 50.0]},
        # {"name": "a_p_sed", "type": "range", "bounds": [0.01, 3.0]},
        # {"name": "rho_sed", "type": "range", "bounds": [1.0, 3.0]},
    ]


STRATEGY_COLORS = {
    "EI": "blue",
    "UCB": "green",
    "Sobol": "red",
    "BAxUS": "orange",
    "Random": "green",
}


@dataclass(frozen=True)
class SWELLEX96Paths:
    path = SWELLEX96PATH
    main_environment_data = SWELLEX96PATH / "env_models" / "main_env.json"
    simple_environment_data = SWELLEX96PATH / "env_models" / "simple_env.json"
    gps_data = SWELLEX96PATH / "gps" / "source_tow.csv"
    acoustic_path = SWELLEX96PATH / "acoustic" / "processed_001"
    outputs = SWELLEX96PATH / "outputs"
    ambiguity_surfaces = SWELLEX96PATH / "acoustic" / "ambiguity_surfaces"
