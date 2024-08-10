#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path

FREQ = [
    148.0,
    235.0,
    388.0,
]
SWELLEX96PATH = Path("../data/swellex96_S5_VLA_inv")

TIME_STEP = 121
TRUE_R = 1.07
TRUE_SRC_Z = 71.11
TILT = 2.0
H_W = 217

TRUE_SIM_VALUES = {
    "rec_r": TRUE_R,
    "src_z": TRUE_SRC_Z,
    "tilt": TILT,
    "h_w": H_W,
    "h_sed": 23.0,
    "c_p_sed_top": 1572.3,
    "dc_p_sed": 20.7,
    "c_p_sed_bot": 1593.0,
    "a_p_sed": 0.2,
    "rho_sed": 1.76,
}

# The following are obtained from running DE with strong priors (narrow bounds).
TRUE_EXP_VALUES = {
    "rec_r": 1.072082,
    "src_z": 71.526224,
    "tilt": 2.067787,
    "h_w": 219.634477,
    "h_sed": 20.070524,
    "c_p_sed_top": 1567.331676,
    "dc_p_sed": 22.206878,
    "c_p_sed_bot": 1589.538554,
    "a_p_sed": 0.652053,
    "rho_sed": 1.990237,
}

VARIABLES = {
    "rec_r": "$r_\mathrm{src}$ [km]",
    "src_z": "$z_\mathrm{src}$ [m]",
    "h_w": "$h_w$ [m]",
    "tilt": "$\\tau$ [$^\circ$]",
    "h_sed": "$h_s$ [m]",
    "c_p_sed_top": "$c_{s,t}$ [m/s]",
    "dc_p_sed": "$\delta c_s$ [m/s]",
    "c_p_sed_bot": "$c_{s,b}$ [m/s]",
    "a_p_sed": "$\\alpha_s$ [dB/kmHz]",
    "rho_sed": "$\\rho_s$ [g/cm$^3$]",
}

SEARCH_SPACE = [
    {"name": "rec_r", "type": "range", "bounds": [TRUE_R - 0.25, TRUE_R + 0.25]},
    {"name": "src_z", "type": "range", "bounds": [60.0, 80.0]},
    {"name": "tilt", "type": "range", "bounds": [-3.0, 3.0]},
    {
        "name": "h_w",
        "type": "range",
        "bounds": [TRUE_SIM_VALUES["h_w"] - 5.0, TRUE_SIM_VALUES["h_w"] + 5.0],
    },
    {"name": "h_sed", "type": "range", "bounds": [15.0, 25.0]},
    {"name": "c_p_sed_top", "type": "range", "bounds": [1560.0, 1580.0]},
    {"name": "dc_p_sed", "type": "range", "bounds": [10.0, 30.0]},
]


STRATEGY_COLORS = {
    "Sobol (100)": "tab:red",
    "Random (100)": "tab:red",
    "Sobol (10k)": "tab:orange",
    "Random (10k)": "tab:orange",
    "BO-UCB": "tab:purple",
    "BO-EI": "tab:blue",
    "BO-LogEI": "tab:green",
    "DE": "black",
}
SORTING_RULE = {
    "Sobol (100)": 0,
    "Random (100)": 1,
    "Sobol (10k)": 2,
    "Random (10k)": 3,
    "BO-UCB": 4,
    "BO-EI": 5,
    "BO-LogEI": 6,
    "DE": 7,
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
