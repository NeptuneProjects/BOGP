# -*- coding: utf-8 -*-

import logging
from pathlib import Path

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