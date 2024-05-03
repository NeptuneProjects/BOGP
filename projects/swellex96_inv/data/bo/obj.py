# -*- coding: utf-8 -*-

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import sys
from typing import Optional

import numpy as np
from tqdm import tqdm
from tritonoa.at.models.kraken.runner import run_kraken
from tritonoa.sp.beamforming import beamformer
from tritonoa.sp.mfp import MatchedFieldProcessor
from tritonoa.sp.processing import simulate_covariance

sys.path.insert(0, str(Path(__file__).parents[4]))
from optimization import utils

import common, param_map

def convert_tensor_to_parameters(x: np.ndarray) -> dict:
    return [
        {d["name"]: float(xs[i]) for i, d in enumerate(common.SEARCH_SPACE)} for xs in x
    ]


def evaluate_objective(objective: callable, parameters: list[dict]) -> list[float]:
    with ProcessPoolExecutor(max_workers=min(len(parameters), 16)) as executor:
        return list(tqdm(executor.map(objective, parameters), total=len(parameters)))


def get_bounds_from_search_space(search_space: list[dict]) -> np.ndarray:
    return np.array([(d["bounds"][0], d["bounds"][1]) for d in search_space])


def get_objective(simulate: bool = True) -> MatchedFieldProcessor:
    if simulate:
        K = simulate_covariance(
            runner=run_kraken,
            parameters=utils.load_env_from_json(common.SWELLEX96Paths.main_environment_data)
            | {
                "rec_r": common.TRUE_VALUES["rec_r"],
                "src_z": common.TRUE_VALUES["src_z"],
                "tilt": common.TRUE_VALUES["tilt"],
            },
            freq=common.FREQ,
        )
    else:
        K = utils.load_covariance_matrices(
            paths=utils.get_covariance_matrix_paths(
                freq=common.FREQ, path=common.SWELLEX96Paths.acoustic_path
            ),
            index=common.TIME_STEP,
        )
    return MatchedFieldProcessor(
        runner=run_kraken,
        covariance_matrix=K,
        freq=common.FREQ,
        parameters=utils.load_env_from_json(common.SWELLEX96Paths.main_environment_data),
        parameter_formatter=param_map.format_parameters,
        beamformer=partial(beamformer, atype="cbf_ml"),
        multifreq_method="product",
    )


def objective(
    x: np.ndarray, true_dim: Optional[int] = None, simulate: bool = True
) -> list[float]:
    """x is assumed to be in [-1, 1]^D"""
    xs = x[:, :true_dim]
    objective = get_objective(simulate=simulate)
    search_space = common.SEARCH_SPACE
    bounds = get_bounds_from_search_space(search_space)
    xs = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * (xs + 1) / 2
    parameters = convert_tensor_to_parameters(xs)
    return evaluate_objective(objective, parameters)
