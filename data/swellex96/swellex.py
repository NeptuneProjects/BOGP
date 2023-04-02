#!/usr/bin/env python3

from pathlib import Path

import numpy as np

from tritonoa.io.profile import read_ssp

HIGH_SIGNAL_TONALS = [
    49.0,
    64.0,
    79.0,
    94.0,
    112.0,
    130.0,
    148.0,
    166.0,
    201.0,
    235.0,
    283.0,
    338.0,
    388.0,
]

# Load CTD data
z_data, c_data, _ = read_ssp(
    Path.cwd() / "Data" / "SWELLEX96" / "CTD" / "i9606.prn", 0, 3, header=None
)
z_data = np.append(z_data, 217).tolist()
c_data = np.append(c_data, c_data[-1]).tolist()

# Define environment parameters
environment = {
    # 1. General
    "title": "SWELLEX96_SIM",
    "model": "KRAKEN",
    # 2. Top medium (halfspace)
    # 3. Layered media
    "layerdata": [
        {"z": z_data, "c_p": c_data, "rho": 1},
        {"z": [217, 240], "c_p": [1572.3, 1593.0], "rho": 1.76, "a_p": 0.2},
        {"z": [240, 1040], "c_p": [1881, 3245], "rho": 2.06, "a_p": 0.06},
    ],
    # 4. Bottom medium
    "bot_opt": "A",
    "bot_c_p": 5200,
    "bot_rho": 2.66,
    "bot_a_p": 0.02,
    # 5. Speed constraints
    "clow": 0,
    "chigh": 1650,
    # 6. Receiver parameters
    "rec_z": np.linspace(94.125, 212.25, 64).tolist(),
    # "tilt": -1
    # 7. Source parameters
    # "rec_r": RANGE_TRUE,
    "src_z": 60.0,
    # "freq": 232.0,
}
