#!/usr/bin/env python3
"""Usage:
python3 ./Source/scripts/configurations.py r
"""
import argparse
from itertools import product
import json
from pathlib import Path

import numpy as np

from tritonoa.io import read_ssp

SMOKE_TEST = False


def format_simstring(simulation_config):
    return " ".join(
        [
            f"{k}={v}" if k != "acq_func" else f"{k}={v['acq_func']}"
            for k, v in simulation_config.items()
        ]
    )


def gen_config_files(simulation):
    # Load CTD data
    z_data, c_data, _ = read_ssp(
        Path.cwd() / "Data" / "SWELLEX96" / "CTD" / "i9606.prn", 0, 3, header=None
    )
    z_data = np.append(z_data, 217).tolist()
    c_data = np.append(c_data, c_data[-1]).tolist()

    # Source parameters
    FREQ = 201
    DEPTH_TRUE = 62
    # RANGE_TRUE = 3

    ## BO-related configurations
    # Set random seed generator
    MAIN_SEED = 2009
    # Number of Monte Carlo simulations to run for each trial
    N_SIM = 500 if not SMOKE_TEST else 2

    # Number of samples used to evaluate & fit surrogate model
    N_SAMPLES = 512
    # Number of random "warmup" samples to initialize BO loop
    N_WARMUP = 10 if not SMOKE_TEST else 2
    # Total number of evaluations (including warmup)
    N_TOTAL = 1000 if not SMOKE_TEST else 10

    # ==========================================================================
    # ============= Edit this section to configure simulations =================
    # ==========================================================================
    # List of acquisition functions to evaluate
    ACQ_FUNCS = [
        {
            "acq_func": "ExpectedImprovement",
            "acq_func_kwargs": {"num_samples": N_SAMPLES},
        },
        {
            "acq_func": "ProbabilityOfImprovement",
            "acq_func_kwargs": {"num_samples": N_SAMPLES},
        },
        {
            "acq_func": "qExpectedImprovement",
            "acq_func_kwargs": {"num_samples": N_SAMPLES},
            "sampler": "SobolQMCNormalSampler",
            "sampler_kwargs": {"num_samples": N_SAMPLES},
        },
    ]

    # List of SNRs to evaluate
    SNR_DB = [np.Inf, 10]

    # List of ranges & depths to simulate
    RANGE = [3.0, 6.0, 10.0, 0.5]
    DEPTH = [62]

    # ==========================================================================
    # ==========================================================================
    # ==========================================================================

    ENVIRONMENT_CONFIG = {
        # General
        "title": "SWELLEX96_SIM",
        "model": "KRAKEN",
        # Top medium
        # Layered media
        "layerdata": [
            {"z": z_data, "c_p": c_data, "rho": 1},
            {"z": [217, 240], "c_p": [1572.37, 1593.02], "rho": 1.8, "a_p": 0.3},
            {"z": [240, 1040], "c_p": [1881, 3245.8], "rho": 2.1, "a_p": 0.09},
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
        # Source parameters
        # "rec_r": RANGE_TRUE,
        "src_z": DEPTH_TRUE,
        "freq": FREQ,
    }

    OPTIMIZER_CONFIG = {
        "n_samples": N_SAMPLES,
        "n_warmup": N_WARMUP,
        "n_total": N_TOTAL,
    }

    fnames = []
    SIMULATION_CONFIGS = []
    if simulation == "range" or simulation == "r":
        qpath = Path.cwd() / "Data" / "Simulations" / "range_estimation" / "queue"
        ENVIRONMENT_CONFIG["src_z"] = DEPTH_TRUE
        SEARCH_CONFIG = [{"name": "rec_r", "bounds": [0.001, 12.0]}]
        OPTIMIZER_CONFIG["search_parameters"] = SEARCH_CONFIG
        for acq_func, snr, rec_r in product(ACQ_FUNCS, SNR_DB, RANGE):
            SIMULATION_CONFIGS.append(
                {"acq_func": acq_func, "snr": snr, "rec_r": rec_r}
            )
    elif simulation == "localize" or simulation == "l":
        qpath = Path.cwd() / "Data" / "Simulations" / "localization" / "queue"
        SEARCH_CONFIG = [
            {"name": "rec_r", "bounds": [0.01, 12.0]},
            {"name": "src_z", "bounds": [0.5, 200.0]},
        ]
        OPTIMIZER_CONFIG["search_parameters"] = SEARCH_CONFIG
        for acq_func, snr, rec_r, src_z in product(ACQ_FUNCS, SNR_DB, RANGE, DEPTH):
            SIMULATION_CONFIGS.append(
                {
                    "acq_func": acq_func,
                    "snr": snr,
                    "rec_r": rec_r,
                    "src_z": src_z,
                }
            )

    for simulation_config in SIMULATION_CONFIGS:
        ENVIRONMENT_CONFIG.update(simulation_config)
        MASTER_CONFIG = {
            "name": "localization",
            "simulation_config": simulation_config,
            "optimizer_config": OPTIMIZER_CONFIG,
            "main_seed": MAIN_SEED,
            "n_sim": N_SIM,
            "environment_config": ENVIRONMENT_CONFIG,
            "desc": format_simstring(simulation_config)
        }
        # simstring = format_simstring(MASTER_CONFIG.get("simulation_config"))
        if not qpath.exists():
            qpath.mkdir(parents=True)
        fname = qpath / f"{MASTER_CONFIG['desc'].replace(' ', '__')}.json"
        with open(fname, "w") as fp:
            json.dump(MASTER_CONFIG, fp, indent=4)
        fnames.append(fname.relative_to(Path.cwd()))
        # count += 1
    return fnames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Specify which simulations to configure."
    )
    parser.add_argument(
        "simulation",
        type=str,
        help="Simulation name.",
        choices=["range", "r", "localize", "l"],
    )
    args = parser.parse_args()
    fnames = gen_config_files(args.simulation)
    # main(args.simulation)
    


# def main(simulation):

#     # Load CTD data
#     z_data, c_data, _ = read_ssp(
#         Path.cwd() / "Data" / "SWELLEX96" / "CTD" / "i9606.prn", 0, 3, header=None
#     )
#     z_data = np.append(z_data, 217).tolist()
#     c_data = np.append(c_data, c_data[-1]).tolist()

#     # Source parameters
#     FREQ = 201
#     DEPTH_TRUE = 62
#     # RANGE_TRUE = 3

#     ## BO-related configurations
#     # Set random seed generator
#     MAIN_SEED = 2009
#     # Number of Monte Carlo simulations to run for each trial
#     N_SIM = 50 if not SMOKE_TEST else 2

#     # Number of samples used to evaluate & fit surrogate model
#     N_SAMPLES = 512
#     # Number of random "warmup" samples to initialize BO loop
#     N_WARMUP = 10
#     # Total number of evaluations (including warmup)
#     N_TOTAL = 503 if not SMOKE_TEST else 10

#     # List of acquisition functions to evaluate
#     ACQ_FUNCS = [
#         {
#             "acq_func": "ExpectedImprovement",
#             "acq_func_kwargs": {"num_samples": N_SAMPLES},
#         },
#         {
#             "acq_func": "ProbabilityOfImprovement",
#             "acq_func_kwargs": {"num_samples": N_SAMPLES},
#         },
#         {
#             "acq_func": "qExpectedImprovement",
#             "acq_func_kwargs": {"num_samples": N_SAMPLES},
#             "sampler": "SobolQMCNormalSampler",
#             "sampler_kwargs": {"num_samples": N_SAMPLES}
#         },
#     ]

#     # List of SNRs to evaluate
#     # SNR_DB = [20]
#     SNR_DB = [np.Inf, 20]

#     ENVIRONMENT_CONFIG = {
#         # General
#         "title": "SWELLEX96_SIM",
#         "model": "KRAKEN",
#         # Top medium
#         # Layered media
#         "layerdata": [
#             {"z": z_data, "c_p": c_data, "rho": 1},
#             {"z": [217, 240], "c_p": [1572.37, 1593.02], "rho": 1.8, "a_p": 0.3},
#             {"z": [240, 1040], "c_p": [1881, 3245.8], "rho": 2.1, "a_p": 0.09},
#         ],
#         # Bottom medium
#         "bot_opt": "A",
#         "bot_c_p": 5200,
#         "bot_rho": 2.7,
#         "bot_a_p": 0.03,
#         # Speed constraints
#         "clow": 0,
#         "chigh": 1600,
#         # Receiver parameters
#         "rec_z": np.linspace(94.125, 212.25, 64).tolist(),
#         # Source parameters
#         # "rec_r": RANGE_TRUE,
#         # "src_z": DEPTH_TRUE,
#         "freq": FREQ,
#     }

#     if simulation == "range" or simulation == "r":
#         print("Configuring range estimation...", end="")
#         ENVIRONMENT_CONFIG["src_z"] = DEPTH_TRUE
#         SEARCH_CONFIG = [{"name": "rec_r", "bounds": [0.001, 12.0]}]

#     elif simulation == "localize" or simulation == "l":
#         print("Configuring localization...", end="")
#         SEARCH_CONFIG = [
#             {"name": "rec_r", "bounds": [0.001, 12.0]},
#             {"name": "src_z", "bounds": [0.5, 200.0]}
#         ]
#     # elif simulation == "geo" or simulation == "g":
#     #     print("Configuring geoacoustic inversion...", end="")
#     # elif simulation == "ssp" or simulation == "s":
#     #     print("Configuring SSP inversion...", end="")

#     SIMULATION_CONFIG = {
#         "acq_func": ACQ_FUNCS,
#         "snr": SNR_DB,
#     }
#     OPTIMIZER_CONFIG = {
#         "search_parameters": SEARCH_CONFIG,
#         "n_samples": N_SAMPLES,
#         "n_warmup": N_WARMUP,
#         "n_total": N_TOTAL,
#         # "q": 3
#     }
#     MASTER_CONFIG = {
#         "simulation_config": SIMULATION_CONFIG,
#         "environment_config": ENVIRONMENT_CONFIG,
#         "optimizer_config": OPTIMIZER_CONFIG,
#         "main_seed": MAIN_SEED,
#         "n_sim": N_SIM,
#     }

#     fpath = Path.cwd() / "Source" / f"config"
#     if not fpath.exists():
#         fpath.mkdir(parents=True)

#     fname = fpath / f"config_{simulation}.json"
#     with open(fname, "w") as fp:
#         json.dump(MASTER_CONFIG, fp, indent=4)

#     print(f"saved to {fname}")
