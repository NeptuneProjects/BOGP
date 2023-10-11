#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from functools import partial
from pathlib import Path
import sys

from ax import Data, Experiment, ParameterType, RangeParameter, SearchSpace
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.modelbridge.registry import Models
from ax.runners.synthetic import SyntheticRunner
from oao.objective import NoiselessFormattedObjective
import torch
from tritonoa.at.models.kraken.runner import run_kraken
from tritonoa.sp.beamforming import beamformer
from tritonoa.sp.mfp import MatchedFieldProcessor
from tritonoa.sp.processing import simulate_covariance

sys.path.insert(0, str(Path(__file__).parents[1]))
from conf import common
from data import formatter

sys.path.insert(0, str(Path(__file__).parents[3]))
from optimization import utils

torch.manual_seed(12345)  # To always get the same Sobol points
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

SIMULATE = False
TIME_STEP = 20
if TIME_STEP == 20:
    TRUE_R = 5.8
    TRUE_SRC_Z = 60
if TIME_STEP == 50:
    TRUE_R = 4.15
    TRUE_SRC_Z = 66
TRUE_TILT = 0.4
TRUE_VALUES = {
    "rec_r": TRUE_R,
    "src_z": TRUE_SRC_Z,
    "c1": 1522.0,
    "dc1": -25.0,
    "dc2": -5.0,
    "dc3": -2.16,
    "dc4": -1.33,
    "dc5": -0.2,
    "dc6": 0.7,
    "dc7": 0.0,
    # "dc8": 0.0,
    # "dc9": 0.0,
    # "dc10": 0.0,
    "h_w": 217.0,
    # "h_s": 23.0,
    # "c_s": 1572.3,
    # "dcdz_s": 0.9,
    # "bot_c_p": 1572.3,
    # "bot_rho": 1.76,
    "tilt": TRUE_TILT,
}


def main():
    base_env = utils.load_env_from_json(common.SWELLEX96Paths.environment_data)

    if SIMULATE:
        K = simulate_covariance(
            runner=run_kraken,
            parameters=base_env
            | {"rec_r": TRUE_R, "src_z": TRUE_SRC_Z, "tilt": TRUE_TILT},
            freq=common.FREQ,
        )
    else:
        K = utils.load_covariance_matrices(
            paths=utils.get_covariance_matrix_paths(
                freq=common.FREQ, path=common.SWELLEX96Paths.acoustic_path
            ),
            index=TIME_STEP,
        )

    processor = MatchedFieldProcessor(
        runner=run_kraken,
        covariance_matrix=utils.load_covariance_matrices(
            paths=utils.get_covariance_matrix_paths(
                freq=common.FREQ, path=common.SWELLEX96Paths.acoustic_path
            ),
            index=TIME_STEP,
        ),
        freq=common.FREQ,
        parameters=base_env,
        parameter_formatter=formatter.format_parameters,
        beamformer=partial(beamformer, atype="cbf_ml"),
        multifreq_method="product",
    )
    objective = NoiselessFormattedObjective(processor, "bartlett", {"minimize": True})


    parameters = [
        {"name": "rec_r", "type": "range", "bounds": [TRUE_R - 0.5, TRUE_R + 0.5]},
        {"name": "src_z", "type": "range", "bounds": [20.0, 100.0]},
        {"name": "c1", "type": "range", "bounds": [1470.0, 1570.0]},
        {"name": "dc1", "type": "range", "bounds": [-40.0, 40.0]},
        {"name": "dc2", "type": "range", "bounds": [-10.0, 10.0]},
        {"name": "dc3", "type": "range", "bounds": [-10.0, 10.0]},
        {"name": "dc4", "type": "range", "bounds": [-5.0, 5.0]},
        {"name": "dc5", "type": "range", "bounds": [-5.0, 5.0]},
        {"name": "dc6", "type": "range", "bounds": [-5.0, 5.0]},
        {"name": "dc7", "type": "range", "bounds": [-5.0, 5.0]},
        # {"name": "dc8", "type": "range", "bounds": [-5.0, 5.0]},
        # {"name": "dc9", "type": "range", "bounds": [-5.0, 5.0]},
        # {"name": "dc10", "type": "range", "bounds": [-10.0, 10.0]},
        {"name": "h_w", "type": "range", "bounds": [180.0, 260.0]},
        # {"name": "h_s", "type": "range", "bounds": [1.0, 100.0]},
        {"name": "bot_c_p", "type": "range", "bounds": [1500.0, 1700.0]},
        # {"name": "bot_rho", "type": "range", "bounds": [1.0, 3.0]},
        # {"name": "dcdz_s", "type": "range", "bounds": [0.0, 3.0]},
        {"name": "tilt", "type": "range", "bounds": [-4.0, 4.0]},
    ]

    search_space = SearchSpace(
        parameters=[
            RangeParameter(
                name=d["name"],
                parameter_type=ParameterType.FLOAT,
                lower=d["bounds"][0],
                upper=d["bounds"][1],
            )
            for d in parameters
        ]
    )

    optimization_config = OptimizationConfig(
        objective=Objective(
            metric=GenericNoisyFunctionMetric(
                name="objective",
                f=processor,
                # param_names=[d["name"] for d in parameters],
                # noise_sd=0.0,  # Set noise_sd=None if you want to learn the noise, otherwise it defaults to 1e-6
            ),
            minimize=True,
        )
    )


    experiment = Experiment(
        name="saasbo_experiment",
        search_space=search_space,
        optimization_config=optimization_config,
        runner=SyntheticRunner(),
    )

    N_INIT = 20
    BATCH_SIZE = 1

    N_BATCHES = 80

    print(f"Doing {N_INIT + N_BATCHES * BATCH_SIZE} evaluations")

    sobol = Models.SOBOL(search_space=experiment.search_space)
    for _ in range(N_INIT):
        experiment.new_trial(sobol.gen(1)).run()

    # Run SAASBO
    data = experiment.fetch_data()
    for i in range(N_BATCHES):
        model = Models.FULLYBAYESIAN(
            experiment=experiment,
            data=data,
            warmup_steps=128,  # Increasing this may result in better model fits
            num_samples=128,  # Increasing this may result in better model fits
            gp_kernel="rbf",  # "rbf" is the default in the paper, but we also support "matern"
            torch_device=tkwargs["device"],
            torch_dtype=tkwargs["dtype"],
            verbose=True,  # Set to True to print stats from MCMC
            disable_progbar=False,  # Set to False to print a progress bar from MCMC
        )
        generator_run = model.gen(BATCH_SIZE)
        trial = experiment.new_batch_trial(generator_run=generator_run)
        trial.run()
        data = Data.from_multiple_data([data, trial.fetch_data()])

        new_value = trial.fetch_data().df["mean"].min()
        print(
            f"Iteration: {i}, Best in iteration {new_value:.3f}, Best so far: {data.df['mean'].min():.3f}"
        )

if __name__ == "__main__":
    main()