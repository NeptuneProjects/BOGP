#!/usr/bin/env python
# -*- coding: utf-8 -*-

from copy import deepcopy
from functools import partial
from pathlib import Path
import sys
from typing import Any, Optional


from ax.models.torch.botorch_modular.optimizer_argparse import optimizer_argparse
from botorch.acquisition.input_constructors import (
    MaybeDict,
    acqf_input_constructor,
    _construct_inputs_mc_base,
    _get_dataset_field,
)
from botorch.models.model import Model
from botorch.utils.datasets import SupervisedDataset
from torch import Tensor

from botorch.acquisition.objective import (
    IdentityMCObjective,
    MCAcquisitionObjective,
    PosteriorTransform,
)

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy,
    qMaxValueEntropy,
)
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
import numpy as np
from oao.objective import NoiselessFormattedObjective
from oao.optimizer import BayesianOptimization
from oao.results import get_results
from oao.space import SearchParameter, SearchSpace
import torch
from tritonoa.at.models.kraken.runner import run_kraken
from tritonoa.sp.beamforming import beamformer
from tritonoa.sp.mfp import MatchedFieldProcessor

sys.path.insert(0, str(Path(__file__).parents[1]))
from conf import common
from data import formatter

sys.path.insert(0, str(Path(__file__).parents[3]))
from optimization import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


@acqf_input_constructor(qLowerBoundMaxValueEntropy)
def construct_inputs_qLBMES(
    model: Model,
    training_data: MaybeDict[SupervisedDataset],
    bounds: list[tuple[float, float]],
    candidate_size: int = 1000,
    maximize: bool = True,
    # TODO: qMES also supports other inputs, such as num_fantasies
) -> dict[str, Any]:
    r"""Construct kwargs for `qLowerBoundMaxValueEntropy` constructor."""

    X = _get_dataset_field(training_data, "X", first_only=True)
    _kw = {"device": X.device, "dtype": X.dtype}
    _rvs = torch.rand(candidate_size, len(bounds), **_kw)
    _bounds = torch.tensor(bounds, **_kw).transpose(0, 1)
    return {
        "model": model,
        "candidate_set": _bounds[0] + (_bounds[1] - _bounds[0]) * _rvs,
        "maximize": maximize,
    }


def main():
    time_step = 20
    base_env = utils.load_env_from_json(common.SWELLEX96Paths.simple_environment_data)

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

    search_space = [
        range_space,
        {"name": "src_z", "type": "range", "bounds": [40.0, 80.0]},
        # {"name": "c1", "type": "range", "bounds": [1500.0, 1550.0]},
        {"name": "dc1", "type": "range", "bounds": [-40.0, 40.0]},
        {"name": "dc2", "type": "range", "bounds": [-10.0, 10.0]},
        {"name": "dc3", "type": "range", "bounds": [-5.0, 5.0]},
        {"name": "dc4", "type": "range", "bounds": [-5.0, 5.0]},
        # {"name": "dc5", "type": "range", "bounds": [-5.0, 5.0]},
        # {"name": "dc6", "type": "range", "bounds": [-5.0, 5.0]},
        # {"name": "dc7", "type": "range", "bounds": [-5.0, 5.0]},
        # {"name": "dc8", "type": "range", "bounds": [-5.0, 5.0]},
        # {"name": "dc9", "type": "range", "bounds": [-5.0, 5.0]},
        # {"name": "dc10", "type": "range", "bounds": [-10.0, 10.0]},
        # {"name": "h_w", "type": "range", "bounds": [200.0, 240.0]},
        # {"name": "h_s", "type": "range", "bounds": [1.0, 100.0]},
        # {"name": "bot_c_p", "type": "range", "bounds": [1500.0, 1700.0]},
        # {"name": "bot_rho", "type": "range", "bounds": [1.0, 3.0]},
        # {"name": "dcdz_s", "type": "range", "bounds": [0.0, 3.0]},
        {"name": "tilt", "type": "range", "bounds": [-3.0, 3.0]},
    ]
    space = SearchSpace([SearchParameter(**d) for d in search_space])

    # model = Models.BOTORCH_MODULAR(
    #     surrogate=Surrogate(
    #         botorch_model_class=FixedNoiseGP,
    #         mll_class=ExactMarginalLogLikelihood,
    #     ),
    #     botorch_acqf_class=qLowerBoundMaxValueEntropy,
    #     acquisition_options={},

    # )

    gs = GenerationStrategy(
        steps=[
            GenerationStep(
                model=Models.SOBOL,
                num_trials=64,
                max_parallelism=64,
            ),
            GenerationStep(
                model=Models.BOTORCH_MODULAR,
                num_trials=500 - 64,
                max_parallelism=None,
                model_kwargs={
                    "surrogate": Surrogate(
                        botorch_model_class=SingleTaskGP,
                        mll_class=ExactMarginalLogLikelihood,
                    ),
                    # "botorch_acqf_class": qMaxValueEntropy,
                    "botorch_acqf_class": qLowerBoundMaxValueEntropy,
                    "torch_device": device,
                },
                model_gen_kwargs={
                    "model_gen_options": {
                        "optimizer_kwargs": {
                            "num_restarts": 120,
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
    )
    opt.run(name="test_bo")
    get_results(
        opt.client,
        times=opt.batch_execution_times,
        minimize=objective.properties.minimize,
    ).to_csv(common.SWELLEX96Paths.outputs / "results.csv")
    opt.client.save_to_json_file(common.SWELLEX96Paths.outputs / "client.json")
    print(opt.client.get_best_parameters(use_model_predictions=False))
    print(opt.client.get_best_parameters(use_model_predictions=True))


if __name__ == "__main__":
    main()
