# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.quasirandom import SobolEngine


def get_random_seeds(n: int) -> list[int]:
    return np.random.choice(np.arange(0, 9999), size=n, replace=False).tolist()


def get_initial_points(
    dim: int, n_pts: int, dtype: torch.dtype, device: torch.device, seed=0
) -> torch.Tensor:
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    # points have to be in [-1, 1]^d
    return 2 * sobol.draw(n=n_pts).to(dtype=dtype, device=device) - 1


def initialize_logger_file(
    fname: Path, logger: logging.Logger, logfmt: logging.Formatter
) -> None:
    fh = logging.FileHandler(filename=fname)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logfmt)
    logger.addHandler(fh)


def initialize_std_logger(logger: logging.Logger, logfmt: logging.Formatter) -> None:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logfmt)
    logger.addHandler(ch)


def get_bounds_from_search_space(search_space: list[dict]) -> np.ndarray:
    return np.array([(d["bounds"][0], d["bounds"][1]) for d in search_space])


def transform_to_original_space(X: np.ndarray, search_space: list[dict]) -> np.ndarray:
    bounds = get_bounds_from_search_space(search_space)
    return bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * (X + 1) / 2


def get_best_params(X: np.ndarray, Y: np.ndarray, search_space: dict) -> np.ndarray:
    best_params = X[np.argmin(Y, axis=0)]
    return transform_to_original_space(
        best_params, search_space
    )


def log_current_value_and_parameters(X: np.ndarray, Y: np.ndarray, search_space: dict, strategy: Optional[str] = None) -> None:
    logging.info(
        f"[Current] Value: {Y[-1].item():.5f} | Parameters: "
        + "".join(
            [
                f"{d['name']}: {v:.2f} | "
                for d, v in zip(
                    search_space,
                    transform_to_original_space(
                        X[-1], search_space
                    ).squeeze(),
                )
            ]
        )[:-3]
    )

def log_best_value_and_parameters(X: np.ndarray, Y: np.ndarray, search_space: dict) -> None:
    best_params = get_best_params(
        X, Y, search_space
    )
    logging.info(
        f"   [Best] Value: {Y.min().item():.5f} | Parameters: "
        + "".join(
            [
                f"{d['name']}: {v:.2f} | "
                for d, v in zip(search_space, best_params.squeeze())
            ]
        )[:-3]
    )
