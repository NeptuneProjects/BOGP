# -*- coding:utf-8 -*-

from dataclasses import dataclass
import logging
import time

import torch
from torch.quasirandom import SobolEngine


@dataclass
class SobolLoopArgs:
    dim: int
    budget: int = 500
    dtype: torch.dtype = torch.double
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loop(
    objective: callable,
    dim: int,
    budget: int,
    dtype: torch.dtype,
    device: torch.device,
    *args,
    **kwargs,
) -> tuple[torch.tensor, torch.tensor]:
    logging.info(f"Running Sobol on {device.type.upper()}.")

    times = []
    start = time.time()

    X = (
        SobolEngine(dim, scramble=True, seed=0)
        .draw(budget)
        .to(dtype=dtype, device=device)
        * 2
        - 1
    )
    Y = torch.tensor(objective(X.detach().cpu().numpy()), dtype=dtype, device=device)

    stop = time.time() - start
    times.append([stop / budget for _ in range(budget)])

    logging.info(
        f"{budget} Sobol trials complete. | Best value: {1 - Y.max().item():.3}"
    )
    return X, Y, times
