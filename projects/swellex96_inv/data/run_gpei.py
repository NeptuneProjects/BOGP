#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
from pathlib import Path
import sys

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
import numpy as np
from oao.objective import NoiselessFormattedObjective
from oao.optimizer import BayesianOptimization
from oao.results import get_results
from oao.space import SearchParameter, SearchSpace
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

NUM_TRIALS = 300
NUM_WARMUP = 200

SIMULATE = False


def monitor(client):
    _, values = client.get_best_parameters(use_model_predictions=False)
    print(
        f"Best observation so far: {values[0]['bartlett']}"
    )


def main():
    base_env = utils.load_env_from_json(common.SWELLEX96Paths.simple_environment_data)

    if SIMULATE:
        K = simulate_covariance(
            runner=run_kraken,
            parameters=base_env
            | {"rec_r": common.TRUE_R, "src_z": common.TRUE_SRC_Z, "tilt": common.TRUE_TILT},
            freq=common.FREQ,
        )
    else:
        K = utils.load_covariance_matrices(
            paths=utils.get_covariance_matrix_paths(
                freq=common.FREQ, path=common.SWELLEX96Paths.acoustic_path
            ),
            index=common.TIME_STEP,
        )

    processor = MatchedFieldProcessor(
        runner=run_kraken,
        covariance_matrix=K,
        freq=common.FREQ,
        parameters=base_env,
        parameter_formatter=formatter.format_parameters,
        beamformer=partial(beamformer, atype="cbf_ml"),
        multifreq_method="product",
    )
    objective = NoiselessFormattedObjective(processor, "bartlett", {"minimize": True})

    search_space = common.SEARCH_SPACE
    space = SearchSpace([SearchParameter(**d) for d in search_space])

    gs = GenerationStrategy(
        steps=[
            GenerationStep(
                model=Models.SOBOL,
                num_trials=NUM_WARMUP,
                max_parallelism=64,
                model_kwargs={"seed": 719}
            ),
            GenerationStep(
                model=Models.GPEI,
                num_trials=NUM_TRIALS - NUM_WARMUP,
                max_parallelism=None,
                model_kwargs={"torch_device": device},
                model_gen_kwargs={
                    "model_gen_options": {
                        "optimizer_kwargs": {
                            "num_restarts": 40,
                            "raw_samples": 4096,
                        }
                    }
                },
            ),
        ]
    )

    opt = BayesianOptimization(
        objective=objective,
        search_space=space,
        strategy=gs,
        monitor=monitor,
    )
    opt.run(name="gpei")
    get_results(
        opt.client,
        times=opt.batch_execution_times,
        minimize=objective.properties.minimize,
    ).to_csv(common.SWELLEX96Paths.outputs / "results_gpei.csv")
    opt.client.save_to_json_file(common.SWELLEX96Paths.outputs / "client_gpei.json")
    print(opt.client.get_best_parameters(use_model_predictions=False))
    print(opt.client.get_best_parameters(use_model_predictions=True))


if __name__ == "__main__":
    main()
