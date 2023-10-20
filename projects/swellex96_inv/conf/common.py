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
    # 148.0,
    # 166.0,
    201.0,
    235.0,
    283.0,
    338.0,
    388.0,
]
SWELLEX96PATH = Path("../data/swellex96_S5_VLA_inv")

TIME_STEP = 20
if TIME_STEP == 20:
    TRUE_R = 5.8
    TRUE_SRC_Z = 60
if TIME_STEP == 50:
    TRUE_R = 4.15
    TRUE_SRC_Z = 66
TRUE_VALUES = {
    "rec_r": TRUE_R,
    "src_z": TRUE_SRC_Z,
    # "c1": 1522.0,
    "dc1": -0.573,
    "dc2": -15.727,
    "dc3": -10.376,
    "dc4": -4.382,
    "dc5": -2.508,
    "h_w": 217.0,
    # "h_s": 23.0,
    # "c_s": 1572.3,
    # "dcdz_s": 0.9,
    # "bot_c_p": 1572.3,
    # "bot_rho": 1.76,
    "tilt": 0.4,
}

VARIABLES = {
    "rec_r": "$r_\mathrm{src}$ [km]",
    "src_z": "$z_\mathrm{src}$ [m]",
    "dc1": "$\Delta c_1$ [m/s]",
    "dc2": "$\Delta c_2$ [m/s]",
    "dc3": "$\Delta c_3$ [m/s]",
    "dc4": "$\Delta c_4$ [m/s]",
    "dc5": "$\Delta c_5$ [m/s]",
    "h_w": "$h_\mathrm{w}$ [m]",
    "bot_c_p": "$c_\mathrm{b}$ [m/s]",
    "tilt": "$\\tau$ [$^\circ$]",
}

SEARCH_SPACE = [
        {"name": "rec_r", "type": "range", "bounds": [TRUE_R-0.5, TRUE_R + 0.5]},
        {"name": "src_z", "type": "range", "bounds": [40.0, 80.0]},
        # {"name": "c1", "type": "range", "bounds": [1470.0, 1570.0]},
        {"name": "dc1", "type": "range", "bounds": [-10.0, 10.0]},
        {"name": "dc2", "type": "range", "bounds": [-40.0, 0.0]},
        {"name": "dc3", "type": "range", "bounds": [-20.0, 0.0]},
        {"name": "dc4", "type": "range", "bounds": [-10.0, 10.0]},
        {"name": "dc5", "type": "range", "bounds": [-5.0, 5.0]},
        {"name": "h_w", "type": "range", "bounds": [TRUE_VALUES["h_w"] - 3.0, TRUE_VALUES["h_w"] + 3.0]},
        # {"name": "h_s", "type": "range", "bounds": [1.0, 100.0]},
        # {"name": "bot_c_p", "type": "range", "bounds": [1560.0, 1600.0]},
        # {"name": "bot_rho", "type": "range", "bounds": [1.0, 3.0]},
        # {"name": "dcdz_s", "type": "range", "bounds": [0.0, 3.0]},
        {"name": "tilt", "type": "range", "bounds": [-2.0, 2.0]},
    ]

@dataclass(frozen=True)
class SWELLEX96Paths:
    path = SWELLEX96PATH
    main_environment_data = SWELLEX96PATH / "env_models" / "main_env.json"
    simple_environment_data = SWELLEX96PATH / "env_models" / "simple_env.json"
    gps_data = SWELLEX96PATH / "gps" / "source_tow.csv"
    acoustic_path = SWELLEX96PATH / "acoustic" / "processed_001"
    outputs = SWELLEX96PATH / "outputs"
    ambiguity_surfaces = SWELLEX96PATH / "acoustic" / "ambiguity_surfaces"
