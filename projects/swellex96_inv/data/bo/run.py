#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from dataclasses import asdict
from enum import Enum
import json
import logging
from pathlib import Path
import random
import sys
import warnings

import numpy as np

import baxus, ei, gibbon, helpers, obj, pi, sobol

sys.path.insert(0, str(Path(__file__).parents[2]))
from conf import common

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


warnings.filterwarnings(
    "ignore", message="A not p.d., added jitter of 1e-08 to the diagonal"
)

TRUE_DIM = len(common.SEARCH_SPACE)


class Strategy(Enum):
    BAXUS = "baxus"
    EI = "ei"
    GIBBON = "gibbon"
    PI = "pi"
    SOBOL = "sobol"
    UCB = "ucb"

    def __str__(self):
        return self.value


def get_loop(
    optim: Strategy, dummy_dim: int, budget: int, n_init: int
) -> tuple[callable, dict]:
    if optim == Strategy.BAXUS:
        loop = baxus.loop
        kwargs = baxus.BAxUSLoopArgs(
            true_dim=TRUE_DIM, budget=budget, n_init=n_init, dummy_dim=dummy_dim
        )
    if optim == Strategy.EI:
        loop = ei.loop
        kwargs = ei.EILoopArgs(dim=TRUE_DIM, budget=budget, n_init=n_init)
    if optim == Strategy.GIBBON:
        loop = gibbon.loop
        kwargs = gibbon.GIBBONLoopArgs(dim=TRUE_DIM, budget=budget, n_init=n_init)
    if optim == Strategy.PI:
        loop = pi.loop
        kwargs = pi.EILoopArgs(dim=TRUE_DIM, budget=budget, n_init=n_init)
    if optim == Strategy.SOBOL:
        loop = sobol.loop
        kwargs = sobol.SobolLoopArgs(dim=TRUE_DIM, budget=budget)
    return loop, asdict(kwargs)


def main(args) -> None:
    loop, kwargs = get_loop(
        args.optim, dummy_dim=args.ndummy, budget=args.budget, n_init=args.init
    )
    dtype = kwargs.pop("dtype", None)
    device = kwargs.pop("device", None)

    random.seed(args.seed)
    seeds = helpers.get_random_seeds(args.runs)

    for seed in seeds:
        serial_name = f"{args.optim}_{seed:04d}_{args.budget}-{args.init}"
        helpers.initialize_loggers(args.dir / f"{serial_name}.log", logger)

        X, Y, times = loop(
            objective=obj.objective, dtype=dtype, device=device, **kwargs
        )

        with open(args.dir / f"{serial_name}.json", "w") as f:
            json.dump(kwargs, f, indent=4)
        
        np.savez(
            args.dir / serial_name,
            X=X.detach().cpu().numpy(),
            Y=Y.detach().cpu().numpy(),
            t=np.array(times),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optim",
        help="Choose an optimization strategy.",
        type=Strategy,
        choices=list(Strategy),
        default="ei",
    )
    parser.add_argument(
        "--budget",
        help="Choose the total budget of trials (including warmup).",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--init",
        help="Choose the number of warmup trials.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--runs",
        help="Specify the number of MC runs for each strategy.",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--seed",
        help="Specify the random seed.",
        type=int,
        default=719,
    )
    parser.add_argument(
        "--dir",
        help="Directory to save outputs.",
        type=Path,
        default=common.SWELLEX96Paths.outputs / "runs",
    )
    parser.add_argument(
        "--ndummy",
        help="[BAxUS only] Specify number of dummy dimensions.",
        type=int,
        default=100,
    )
    args = parser.parse_args()
    main(args)
