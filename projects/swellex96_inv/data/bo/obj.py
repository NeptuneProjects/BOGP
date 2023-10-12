# -*- coding: utf-8 -*-

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import sys
from typing import Optional

import numpy as np
from tritonoa.at.models.kraken.runner import run_kraken
from tritonoa.sp.beamforming import beamformer
from tritonoa.sp.mfp import MatchedFieldProcessor

sys.path.insert(0, str(Path(__file__).parents[2]))
from conf import common
from data import formatter

sys.path.insert(0, str(Path(__file__).parents[4]))
from optimization import utils


def convert_tensor_to_parameters(x: np.ndarray) -> dict:
    return [
        {d["name"]: float(xs[i]) for i, d in enumerate(common.SEARCH_SPACE)} for xs in x
    ]


def evaluate_objective(objective: callable, parameters: list[dict]) -> list[float]:
    with ThreadPoolExecutor(max_workers=min(len(parameters), 64)) as executor:
        results = executor.map(objective, parameters)
    return [1 - result for result in results]


def get_bounds_from_search_space(search_space: list[dict]) -> np.ndarray:
    return np.array([(d["bounds"][0], d["bounds"][1]) for d in search_space])


def get_objective() -> MatchedFieldProcessor:
    base_env = utils.load_env_from_json(common.SWELLEX96Paths.simple_environment_data)
    return MatchedFieldProcessor(
        runner=run_kraken,
        covariance_matrix=utils.load_covariance_matrices(
            paths=utils.get_covariance_matrix_paths(
                freq=common.FREQ, path=common.SWELLEX96Paths.acoustic_path
            ),
            index=common.TIME_STEP,
        ),
        freq=common.FREQ,
        parameters=base_env,
        parameter_formatter=formatter.format_parameters,
        beamformer=partial(beamformer, atype="cbf_ml"),
        multifreq_method="product",
    )


def objective(x: np.ndarray, true_dim: Optional[int] = None) -> list[float]:
    """x is assumed to be in [-1, 1]^D"""
    xs = x[:, :true_dim]
    objective = get_objective()
    search_space = common.SEARCH_SPACE
    bounds = get_bounds_from_search_space(search_space)
    xs = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * (xs + 1) / 2
    parameters = convert_tensor_to_parameters(xs)
    return evaluate_objective(objective, parameters)
