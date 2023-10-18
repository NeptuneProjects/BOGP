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
    # with ThreadPoolExecutor(max_workers=16) as executor:
    #     futures = [
    #         executor.submit(objective, x[None, :])
    #         for x in tqdm(X, total=len(X), desc="Evaluating points", bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}")
    #     ]
    #     Y = np.array([f.result() for f in futures])
    
    end = time.time() - start
    times = [(i + 1) * end / len(X) for i in range(len(X))]

    logging.info(f"Trial {len(X)} | Best value: {1 - Y.max().item():.3}")

    return X, Y, times
