# -*- coding:utf-8 -*-

from dataclasses import dataclass
import logging
import time

import torch


@dataclass
class RandomLoopArgs:
    dim: int
    budget: int = 500
    seed: int = None
    dtype: torch.dtype = torch.double
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loop(
    objective: callable,
    dim: int,
    budget: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 0,
    *args,
    **kwargs,
) -> tuple[torch.tensor, torch.tensor]:
    logging.info(f"Running random on {device.type.upper()}.")

    start = time.time()

    X = torch.rand(budget, dim, dtype=dtype, device=device) * 2 - 1
    Y = torch.tensor(objective(X.detach().cpu().numpy()), dtype=dtype, device=device)

    stop = time.time() - start
    times = [stop / budget for _ in range(budget)]

    return X, Y, times
