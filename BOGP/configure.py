#!/usr/bin/env python3

from datetime import datetime
from itertools import product
import os
from pathlib import Path
import random
from typing import Union

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.service.ax_client import ObjectiveProperties
from botorch.acquisition import (
    qUpperConfidenceBound,
    qExpectedImprovement,
    qProbabilityOfImprovement,
    ProbabilityOfImprovement,
)
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
import numpy as np

from oao.common import UNINFORMED_STRATEGIES
from oao.utilities import save_config
import swellex
from tritonoa.kraken import run_kraken
from tritonoa.sp import added_wng, snrdb_to_sigma

NUM_MC_RUNS = 1
NUM_RESTARTS = 20
NUM_SAMPLES = 512
NUM_TRIALS = 200
NUM_WARMUP = 10
Q = 5
ROOT = Path.cwd() / "Data"
SEED = 2009
SERIAL = datetime.now().strftime("serial_%Y%m%dT%H%M%S")


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
                scenario_folder = mode_folder / scenario_name

                if mode["mode"] == "simulation":

                    data_folder = scenario_folder / "data"
                    os.makedirs(data_folder)

                    K = simulate_measurement_covariance(
                        mode["obj_func_parameters"]["env_parameters"]
                        | scenario
                        | {"tmpdir": data_folder}
                    )
                    np.save(data_folder / "measurement_covariance.npy", K)

                param_names = [
                    d["name"] for d in mode["experiment_kwargs"]["parameters"]
                ]
                for k in scenario.keys():
                    if k not in param_names:
                        mode["obj_func_parameters"]["env_parameters"][k] = scenario[k]

                # rand/sobol/grid/bo/etc.
                for strategy in mode["strategies"]:

                    if strategy["loop_type"] not in UNINFORMED_STRATEGIES:
                        if (
                            strategy["generation_strategy"]
                            ._steps[1]
                            .model_kwargs["botorch_acqf_class"]
                            .__name__
                            == "qExpectedImprovement"
                        ):
                            acqf = "_qei"
                        elif (
                            strategy["generation_strategy"]
                            ._steps[1]
                            .model_kwargs["botorch_acqf_class"]
                            .__name__
                            == "qProbabilityOfImprovement"
                        ):
                            acqf = "_qpi"
                        elif (
                            strategy["generation_strategy"]
                            ._steps[1]
                            .model_kwargs["botorch_acqf_class"]
                            .__name__
                            == "qUpperConfidenceBound"
                        ):
                            acqf = "_qucb"
                    else:
                        acqf = ""

                    # trial seed
                    for seed in mc_seeds:

                        run_config = {
                            "experiment_kwargs": mode["experiment_kwargs"],
                            "seed": seed,
                            "strategy": strategy,
                            "obj_func_parameters": mode["obj_func_parameters"],
                            "evaluation_config": mode["evaluation_config"],
                            "destination": str(
                                scenario_folder.relative_to(mode_folder)
                            ),
                        }

                        config_name = f"config__{scenario_name}__{strategy['loop_type'] + acqf}__{seed:010d}.json"
                        save_config(q_folder / config_name, run_config)


def simulate_measurement_covariance(env_parameters: dict) -> np.array:
    snr = env_parameters.pop("snr")
    sigma = snrdb_to_sigma(snr)
    p = run_kraken(env_parameters)
    p /= np.linalg.norm(p)
    noise = added_wng(p.shape, sigma=sigma, cmplx=True)
    p += noise
    K = p.dot(p.conj().T)
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


if __name__ == "__main__":

    optimizations = [
        {
            "type": "range_estimation",
            "modes": [
                {
                    "mode": "simulation",
                    "serial": SERIAL,
                    "scenarios": {
                        "rec_r": [1.0, 3.0, 5.0, 7.0],
                        "src_z": [62.0],
                        "snr": [20],
                    },
                    "strategies": [
                        {"loop_type": "grid", "num_trials": 10},
                        {"loop_type": "lhs", "num_trials": NUM_TRIALS},
                        {"loop_type": "random", "num_trials": NUM_TRIALS},
                        {"loop_type": "sobol", "num_trials": NUM_TRIALS},
                        {
                            "loop_type": "greedy_batch",  # <--- This is where to specify loop,
                            "num_trials": NUM_TRIALS,
                            "batch_size": Q,
                            "generation_strategy": GenerationStrategy(
                                [
                                    GenerationStep(
                                        model=Models.SOBOL,
                                        num_trials=NUM_WARMUP,
                                        max_parallelism=NUM_WARMUP,
                                        model_kwargs={"seed": SEED},
                                    ),
                                    GenerationStep(
                                        model=Models.BOTORCH_MODULAR,
                                        num_trials=NUM_TRIALS - NUM_WARMUP,
                                        max_parallelism=None,
                                        model_kwargs={
                                            "surrogate": Surrogate(
                                                botorch_model_class=SingleTaskGP,
                                                mll_class=ExactMarginalLogLikelihood,
                                            ),
                                            "botorch_acqf_class": qExpectedImprovement,
                                            # "torch_device": DEVICE,
                                        },
                                        model_gen_kwargs={
                                            "model_gen_options": {
                                                "optimizer_kwargs": {
                                                    "num_restarts": NUM_RESTARTS,
                                                    "raw_samples": NUM_SAMPLES,
                                                }
                                            }
                                        },
                                    ),
                                ]
                            ),
                        },
                    ],
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
