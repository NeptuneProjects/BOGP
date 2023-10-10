#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from functools import partial
from pathlib import Path
import sys

from ax import Data, Experiment, ParameterType, RangeParameter, SearchSpace
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.metrics.branin import BraninMetric
from ax.metrics.noisy_function import GenericNoisyFunctionMetric
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.runners.synthetic import SyntheticRunner
from oao.objective import NoiselessFormattedObjective
from oao.optimizer import BayesianOptimization
from oao.results import get_results

# from oao.space import SearchParameter, SearchSpace
import torch
from tritonoa.at.models.kraken.runner import run_kraken
from tritonoa.sp.beamforming import beamformer
from tritonoa.sp.mfp import MatchedFieldProcessor

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

time_step = 20


def main():
    base_env = utils.load_env_from_json(common.SWELLEX96Paths.environment_data)

    processor = MatchedFieldProcessor(
        runner=run_kraken,
        covariance_matrix=utils.load_covariance_matrices(
            paths=utils.get_covariance_matrix_paths(
                freq=common.FREQ, path=common.SWELLEX96Paths.acoustic_path
            ),
            index=time_step,
        ),
        freq=common.FREQ,
        parameters=base_env,
        parameter_formatter=formatter.format_parameters,
        beamformer=partial(beamformer, atype="cbf_ml"),
        multifreq_method="product",
    )
    objective = NoiselessFormattedObjective(processor, "bartlett", {"minimize": True})

    if time_step == 20:
        range_space = {"name": "rec_r", "type": "range", "bounds": [5.5, 6.0]}
    if time_step == 50:
        range_space = {"name": "rec_r", "type": "range", "bounds": [3.8, 4.3]}

    parameters = [
        range_space,
        {"name": "src_z", "type": "range", "bounds": [40.0, 80.0]},
        {"name": "c1", "type": "range", "bounds": [1500.0, 1550.0]},
        {"name": "dc1", "type": "range", "bounds": [-40.0, 40.0]},
        {"name": "dc2", "type": "range", "bounds": [-10.0, 10.0]},
        {"name": "dc3", "type": "range", "bounds": [-5.0, 5.0]},
        {"name": "dc4", "type": "range", "bounds": [-5.0, 5.0]},
        {"name": "dc5", "type": "range", "bounds": [-5.0, 5.0]},
        {"name": "dc6", "type": "range", "bounds": [-5.0, 5.0]},
        {"name": "dc7", "type": "range", "bounds": [-5.0, 5.0]},
        {"name": "dc8", "type": "range", "bounds": [-5.0, 5.0]},
        {"name": "dc9", "type": "range", "bounds": [-5.0, 5.0]},
        # {"name": "dc10", "type": "range", "bounds": [-10.0, 10.0]},
        {"name": "h_w", "type": "range", "bounds": [200.0, 240.0]},
        # {"name": "h_s", "type": "range", "bounds": [1.0, 100.0]},
        {"name": "bot_c_p", "type": "range", "bounds": [1500.0, 1700.0]},
        # {"name": "bot_rho", "type": "range", "bounds": [1.0, 3.0]},
        # {"name": "dcdz_s", "type": "range", "bounds": [0.0, 3.0]},
        {"name": "tilt", "type": "range", "bounds": [-3.0, 3.0]},
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

    N_INIT = 32
    BATCH_SIZE = 1

    N_BATCHES = 48

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