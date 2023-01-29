#!/usr/bin/env python3

from pathlib import Path

import numpy as np

from tritonoa.io import read_ssp


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
        {"z": [217, 240], "c_p": [1572.37, 1593.02], "rho": 1.8, "a_p": 0.3},
        {"z": [240, 1040], "c_p": [1881, 3245.8], "rho": 2.1, "a_p": 0.09},
    ],
    # 4. Bottom medium
    "bot_opt": "A",
    "bot_c_p": 5200,
    "bot_rho": 2.7,
    "bot_a_p": 0.03,
    # 5. Speed constraints
    "clow": 0,
    "chigh": 1600,
    # 6. Receiver parameters
    "rec_z": np.linspace(94.125, 212.25, 64).tolist(),
    # 7. Source parameters
    # "rec_r": RANGE_TRUE,
    "src_z": 62.0,
    "freq": 232.0,
}
