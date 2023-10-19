# -*- coding: utf-8 -*-

from dataclasses import dataclass
import logging
import time

import torch


@dataclass
class GridLoopArgs:
    dim: int
    budget: int = 3


def loop(
    objective: callable,
    dim,
    budget,
    *args,
    **kwargs,
):
    logging.info(f"Running grid search.")

    points = torch.linspace(-1, 1, budget)
    X = torch.cartesian_prod(*[points for _ in range(dim)])

    start = time.time()

    Y = torch.tensor(objective(X.detach().cpu().numpy()))

    end = time.time() - start
    times = [(i + 1) * end / len(X) for i in range(len(X))]

    return X, Y, times
