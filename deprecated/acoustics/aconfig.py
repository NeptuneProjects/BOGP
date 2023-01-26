#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, "/Users/williamjenkins/Research/Code/TritonOA/")
sys.path.insert(0, "/Users/williamjenkins/Research/Projects/BOGP/Source/")
from tritonoa.io import read_ssp
from BOGP.utils import write_config


def configure(path_base=Path.cwd(), path_scene=Path.cwd()):
    cwd = Path.cwd()
    # ==================================================================
    # ======================== BASE ENVIRONMENT ========================
    # ==================================================================
    parameters = dict()
    # ------------------------- Top Parameters -------------------------
    # None to configure.

    # ------------------------ Layer Parameters ------------------------
    fname = Path(
        "/Users/williamjenkins/Research/Projects/BOGP/Data/SWELLEX96/CTD/i9606.prn"
    )
    z_data, c_data, _ = read_ssp(fname, 0, 3, header=None)
    z_data = np.append(z_data, 217).tolist()
    c_data = np.append(c_data, c_data[-1]).tolist()

    parameters["layerdata"] = [
        {"z": z_data, "c_p": c_data, "rho": 1},
        {"z": [217, 240], "c_p": [1572.37, 1593.02], "rho": 1.8, "a_p": 0.3},
        {"z": [240, 1040], "c_p": [1881, 3245.8], "rho": 2.1, "a_p": 0.09},
    ]

    # ----------------------- Bottom Parameters ------------------------
    # Note: By default, bottom depth is pulled from layer data
    parameters["bot_opt"] = "A"
    parameters["bot_c_p"] = 5200
    parameters["bot_rho"] = 2.7
    parameters["bot_a_p"] = 0.03

    # ----------------------- Source Parameters ------------------------
    # None to configure.

    # ---------------------- Receiver Parameters ----------------------
    parameters["rec_z"] = np.linspace(94.125, 212.25, 64).tolist()

    # ---------------------- Freq/Mode Parameters ----------------------
    parameters["freq"] = 148
    parameters["clow"] = 0
    parameters["chigh"] = 1600

    write_config(path_base, parameters)

    # ==================================================================
    # ======================= SCENARIO PARAMETERS ======================
    # ==================================================================
    parameters = dict()
    # ----------------------- Source Parameters ------------------------
    depth_true = 54
    parameters["src_z"] = depth_true

    # ---------------------- Receiver Parameters ----------------------
    range_true = 4.4
    parameters["rec_r"] = range_true

    # -------------------- Miscellaneous Parameters --------------------
    parameters["title"] = "KRAKENC"
    parameters[
        "tmpdir"
    ] = str((cwd / "Data" / "Experiments" / "Localization2D" / "Simulated" / "Scene001" / "ReceivedData").relative_to(cwd))
    parameters["model"] = "KRAKENC"

    write_config(path_scene, parameters)


if __name__ == "__main__":
    configure()
