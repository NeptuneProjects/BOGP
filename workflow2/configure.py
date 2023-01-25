#!/usr/bin/env python3

from datetime import datetime
from itertools import product
import os
from pathlib import Path
import random
from typing import Union

from ax.service.ax_client import ObjectiveProperties
import numpy as np

from oao.utilities import save_config
import swellex
from tritonoa.kraken import run_kraken
from tritonoa.sp import added_wng, snrdb_to_sigma


NUM_MC_RUNS = 1
NUM_RESTARTS = 40
NUM_SAMPLES = 1024
NUM_TRIALS = 100
NUM_WARMUP = 10
ROOT = Path.cwd().parent / "Data" / "workflow2"
SEED = 2009
SERIAL = datetime.now().strftime("serial_%Y%m%dT%H%M%S")

print(Path.cwd())


def create_config_files(config: dict):

    # range estimation or localization
    for optim in config:

        # simulation or experimental
        for mode in optim["modes"]:
            mode_folder = ROOT / optim["type"] / mode["mode"] / mode["serial"]
            q_folder = mode_folder / "queue"
            os.makedirs(mode_folder)
            os.mkdir(q_folder)

            mc_seeds = get_mc_seeds(mode["main_seed"], mode["num_runs"])

            # range/depth/snr/etc.
            scenarios = get_scenario_dict(**mode["scenarios"])
            for scenario in scenarios:
                scenario_name = get_scenario_path(scenario)

                if mode["mode"] == "simulation":
                    data_folder = mode_folder / scenario_name / "data"
                    os.makedirs(data_folder)
                    K = simulate_measurement_covariance(
                        mode["obj_func_parameters"]["env_parameters"] | scenario
                    )
                    np.save(data_folder / "measurement_covariance.npy", K)

                # rand/sobol/grid/bo/etc.
                for strategy in mode["strategies"]:

                    # trial seed
                    for seed in mc_seeds:



                        run_config = {
                            "experiment_kwargs": mode["experiment_kwargs"],
                            "obj_func_parameters": mode["obj_func_parameters"],
                            "num_trials": mode["num_trials"]
                        }

                        

                        config_name = (
                            f"config_{scenario_name}_{strategy}_{seed:010d}.json"
                        )
                        save_config(q_folder / config_name, run_config)


def simulate_measurement_covariance(env_parameters):
    snr = env_parameters.pop("snr")
    sigma = snrdb_to_sigma(snr)
    p = run_kraken(env_parameters)
    p /= np.linalg.norm(p)
    noise = added_wng(p.shape, sigma=sigma, cmplx=True)
    p += noise
    K = p.dot(p.conj().T)
    clean_up_kraken_files(Path.cwd())
    return K


def clean_up_kraken_files(path: Union[Path, str]):
    if isinstance(path, str):
        path = Path(path)
    extensions = ["env", "mod", "prt"]
    [[f.unlink() for f in path.glob(f"*.{ext}")] for ext in extensions]


def get_scenario_dict(**kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for i in product(*vals):
        yield dict(zip(keys, i))


def get_scenario_path(scenario: dict) -> str:
    return "__".join([f"{k}={v}" for k, v in scenario.items()])


def get_mc_seeds(main_seed: int, num_runs: int) -> list:
    random.seed(main_seed)
    return [random.randint(0, int(1e9)) for _ in range(num_runs)]


optimizations = [
    {
        "type": "range_estimation",
        "modes": [
            {
                "mode": "simulation",
                "serial": SERIAL,
                "scenarios": {
                    "rec_r": [1.0, 3.0, 5.0, 7.0],
                    "src_z": [56.0, 62.0],
                    "snr": [20],
                },
                "strategies": {
                    "grid": {"num_trials": [10, 10]},
                    "lhs": {"num_trials": NUM_TRIALS},
                    "random": {"num_trials": NUM_TRIALS},
                    "sobol": {"num_trials": NUM_TRIALS},
                },
                "experiment_kwargs": {
                    "name": "mfp_test",
                    "parameters": [
                        {
                            "name": "rec_r",
                            "type": "range",
                            "bounds": [0.1, 10.0],
                            "value_type": "float",
                            "log_scale": False,
                        },
                    ],
                    "objectives": {"bartlett": ObjectiveProperties(minimize=False)},
                },
                "obj_func_parameters": {"env_parameters": swellex.environment},
                "main_seed": SEED,
                "num_runs": NUM_MC_RUNS,
                "num_trials": NUM_TRIALS,
                "evaluation_config": None,
            },
            # {"mode": "experimental", "serial": serial},
        ],
    },
    # {
    #     "type": "localization",
    #     "modes": None
    # }
]

create_config_files(optimizations)
