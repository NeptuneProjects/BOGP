#!/usr/bin/env python3

import sys

import numpy as np

sys.path.insert(0, '/Users/williamjenkins/Research/Code/TritonOA/')
from tritonoa.io import read_ssp

# Load CTD data
z_data, c_data, _ = read_ssp(ROOT / "Data" / "SWELLEX96" / "CTD" / "i9606.prn", 0, 3, header=None)
z_data = np.append(z_data, 217).tolist()
c_data = np.append(c_data, c_data[-1]).tolist()

# SWELLEX-96
ENV_SWELLEX_96 = {
    # General
    "title": "SWELLEX96_SIM",
    # "tmpdir": "tmp",
    # "model": "KRAKENC",
    # Top medium
    # Layered media
    "layerdata": [
        {
            "z": z_data,
            "c_p": c_data,
            "rho": 1
        },
        {
            "z": [217, 240],
            "c_p": [1572.37, 1593.02],
            "rho": 1.8,
            "a_p": 0.3
        },
        {
            "z": [240, 1040],
            "c_p": [1881, 3245.8],
            "rho": 2.1,
            "a_p": 0.09
        }
    ],
    # Bottom medium
    "bot_opt": "A",
    "bot_c_p": 5200,
    "bot_rho": 2.7,
    "bot_a_p": 0.03,
    # Speed constraints
    "clow": 0,
    "chigh": 1600,
    # Receiver parameters
    "rec_z": np.linspace(94.125, 212.25, 64).tolist(),
    # # Source parameters
    # "rec_r": range_true,
    # "src_z": depth_true,
    # "freq": freq
}