#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from pathlib import Path
import random
from typing import Optional, Union

from hydra.core.hydra_config import HydraConfig
import numpy as np
from oao.space import SearchParameter, SearchSpace, SearchSpaceBounds
from tritonoa.sp.beamforming import covariance

from optimization.parameterization import Parameterization


def load_env_from_json(path: os.PathLike) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def simulate_covariance(
    runner: callable, parameters: Union[dict, list[dict]], freq: list
) -> np.ndarray:
    # TODO: Migrate this function to TritonOA.
    K = []
    for f in freq:
        p = runner(parameters | {"freq": f})
        p /= np.linalg.norm(p)
        K.append(covariance(p))
    return np.array(K)


def save_covariance_data(covariance_matrix: np.ndarray, path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    savepath = path / "covariance.npy"
    np.save(savepath, covariance_matrix)
    return savepath


def load_covariance_matrices(paths: list, index: Optional[int] = None) -> np.ndarray:
    K = []
    print(paths)
    for path in paths:
        K.append(np.load(Path(path)))
    if index is None:
        return np.array(K)
    return np.array(K)[:, index, ...]


def get_covariance_index_from_parameterization(key: str, parameterization: dict) -> int:
    return parameterization[key]


def get_covariance_matrix_paths(freq: list, path: Path) -> list[Path]:
    return [path / f"{f:.1f}Hz" / "covariance.npy" for f in freq]


def adjust_bounds(lower: float, lower_bound: float, upper: float, upper_bound: float) -> float:
    if lower < lower_bound:
        new_lower = lower_bound
        new_upper = new_lower + abs(lower) + abs(upper)
    elif upper > upper_bound:
        new_upper = upper_bound
        new_lower = new_upper - abs(lower) - abs(upper)
    else:
        new_lower = lower
        new_upper = upper
    return new_lower, new_upper


def format_search_space(search_space: SearchSpaceBounds, scenario: dict) -> dict:
    parameters = []
    for bounds in search_space.bounds:
        if bounds.name in scenario.keys():
            lower = scenario[bounds.name] + bounds.lower_bound
            upper = scenario[bounds.name] + bounds.upper_bound
            if lower < bounds.min_lower_bound:
                lower = bounds.min_lower_bound
                upper = lower + abs(bounds.lower_bound) + abs(bounds.upper_bound)
            elif upper > bounds.max_upper_bound:
                upper = bounds.max_upper_bound
                lower = upper - abs(bounds.upper_bound) - abs(bounds.lower_bound)

        parameters.append(
            {
                "name": bounds.name,
                "type": "range",
                "bounds": [float(lower), float(upper)],
            }
        )

    return SearchSpace([SearchParameter(**d) for d in parameters])


def get_random_seeds(main_seed: int, num_mc_runs: int) -> list[int]:
    random.seed(main_seed)
    return [random.randint(0, int(1e9)) for _ in range(num_mc_runs)]


def check_serial_exists(path: Path) -> bool:
    if path.exists():
        while True:
            overwrite = input(f"{path} already exists. Overwrite? [y/n] ")
            if overwrite == "y":
                return True
            if overwrite == "n":
                return False


def make_serial_paths(path: Path) -> Path:
    qpath = path / "queue"
    qpath.mkdir(parents=True, exist_ok=check_serial_exists(path))
    return qpath


def make_save_path(path: Path, scenario: dict, strategy: str, seed: int) -> Path:
    savepath = path / format_scenario_path(scenario) / strategy / f"{int(seed):09d}"
    savepath.mkdir(parents=True, exist_ok=True)
    return savepath


def format_scenario_path(scenario: dict) -> str:
    return "".join(
        [
            f"{k}={v:{'.2f' if isinstance(v, float) else ''}}__"
            for k, v in scenario.items()
        ]
    )[:-2]


def get_scenario_for_savepath(scenario: dict, keys: list) -> str:
    return {k: v for k, v in scenario.items() if k in keys}


def get_scenario_keys_for_savepath(
    parameterization: Parameterization, search_space=None
) -> list:
    return list(
        set(parameterization.indexed.scenario.keys())
        | set(parameterization.permuted.scenario.keys())
        | set([p.name for p in search_space.bounds])
    )


def setup_scenario_path(scenario, scenario_keys, savepath_base):
    savepath_scenario = get_scenario_for_savepath(scenario, scenario_keys)
    path = savepath_base / format_scenario_path(savepath_scenario)
    return path


def format_configfile_name(path: Path, optimizer: str, seed: int):
    return f"{path.parents[1].name}__{optimizer}__{int(seed):09d}.yaml"


def get_config_path() -> Path:
    hydra_conf_path = Path(HydraConfig.get().runtime.config_sources[1].path)
    hydra_conf_name = HydraConfig.get().job.config_name
    return hydra_conf_path / hydra_conf_name


def move_config_to_savepath(savepath: Path) -> None:
    config_path = get_config_path()
    config_path.rename(savepath / config_path.name)
