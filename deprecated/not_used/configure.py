from pathlib import Path

import numpy as np

import utils

# ======================================================================
# ======================== EXPERIMENT PARAMETERS =======================
# ======================================================================

expparams = {
    "experiments": ["Localization2D", "Inversion"],
    "method": ["BOGP"],
    "acqfunc": ["PI", "EI"],
    "kernel": ["RBF", "MAT", "PER", "COS"]
}

# ======================================================================
# ========================== FIXED PARAMETERS ==========================
# ======================================================================

# ---------------------- Miscellaneous Parameters ----------------------
fixedparams = dict()
fixedparams["title"] = "SWELLEX96"
fixedparams["tmpdir"] = "tmp"
fixedparams["model"] = "KRAKENC"

# --------------------------- Top Parameters ---------------------------
# None to configure.

# -------------------------- Layer Parameters --------------------------
fixedparams["layerdata"] = [
    # {
    #     "z": z_data,
    #     "c_p": c_data,
    #     "rho": 1
    # },
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
]

# ------------------------- Bottom Parameters --------------------------
# Note: By default, bottom depth is pulled from layer data
fixedparams["bot_opt"] = "A"
fixedparams["bot_c_p"] = 5200.
fixedparams["bot_rho"] = 2.7
fixedparams["bot_a_p"] = 0.03

# ------------------------- Source Parameters --------------------------
# fixedparams["src_z"] = depth_true

# ------------------------ Receiver Parameters ------------------------
fixedparams["rec_z"] = np.linspace(94.125, 212.25, 64).tolist()
# fixedparams["rec_r"] = range_true


# ------------------------ Freq/Mode Parameters ------------------------
# fixedparams["freq"] = freq
fixedparams["clow"] = 0
fixedparams["chigh"] = 1600


# ======================================================================
# ============================ SEARCH SPACE ============================
# ======================================================================
searchparams = [
    {
        "name": "rec_r",
        "type": "range",
        "bounds": [3.0, 6.0],
        "value_type": "float"
    },
    {
        "name": "src_z",
        "type": "range",
        "bounds": [10., 100.],
        "value_type": "float"
    }
]


def main(path=Path.cwd()):
    exp = utils.Experiment(expparams, fixedparams, searchparams)
    exp.format_exp_names()
    exp.write_config(path=path)
    return exp.experiments, exp.expparams, exp.fixedparams, exp.searchparams


if __name__ == "__main__":
    main()
