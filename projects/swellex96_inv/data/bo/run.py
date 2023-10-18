#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from dataclasses import asdict
from enum import Enum
from functools import partial
import json
import logging
from pathlib import Path
import random
import sys
import warnings

import numpy as np

import baxus, ei, gibbon, helpers, obj, pi, sobol, ucb

sys.path.insert(0, str(Path(__file__).parents[2]))
from conf import common

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logfmt = logging.Formatter("[%(levelname)s][%(asctime)s] %(message)s")
helpers.initialize_std_logger(logger, logfmt)

warnings.filterwarnings(
    "ignore", message="A not p.d., added jitter of 1e-08 to the diagonal"
)

SMOKE_TEST = False
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


def get_bounds_from_search_space(search_space: list[dict]) -> np.ndarray:
    return np.array([(d["bounds"][0], d["bounds"][1]) for d in search_space])


def transform_to_original_space(X: np.ndarray, search_space: list[dict]) -> np.ndarray:
    bounds = get_bounds_from_search_space(search_space)
    return bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * (X + 1) / 2


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
        kwargs = pi.PILoopArgs(dim=TRUE_DIM, budget=budget, n_init=n_init)
    if optim == Strategy.SOBOL:
        loop = sobol.loop
        kwargs = sobol.SobolLoopArgs(dim=TRUE_DIM, budget=budget)
    if optim == Strategy.UCB:
        loop = ucb.loop
        kwargs = ucb.UCBLoopArgs(dim=TRUE_DIM, budget=budget, n_init=n_init)
    return loop, asdict(kwargs)


def main(args) -> None:
    loop, kwargs = get_loop(
        args.optim, dummy_dim=args.ndummy, budget=args.budget, n_init=args.init
    )
    dtype = kwargs.pop("dtype", None)
    device = kwargs.pop("device", None)

    random.seed(args.seed)
    seeds = helpers.get_random_seeds(args.runs)

    print("_" * 80)
    for seed in seeds:
        serial_name = f"{args.optim}_{args.budget}-{args.init}_{seed:04d}"
        helpers.initialize_logger_file(args.dir / f"{serial_name}.log", logger, logfmt)

        X, Y, times = loop(
            objective=partial(obj.objective, simulate=args.simulate),
            dtype=dtype,
            device=device,
            **{**kwargs | {"seed": seed}},
        )
        X, Y, times = (
            X.detach().cpu().numpy(),
            Y.detach().cpu().numpy(),
            np.array(times),
        )

        with open(args.dir / f"{serial_name}.json", "w") as f:
            json.dump(kwargs, f, indent=4)

        np.savez(
            args.dir / serial_name,
            X=X,
            Y=Y,
            t=times,
        )

        best_params = X[np.argmin(Y, axis=0)]
        best_params = transform_to_original_space(best_params, common.SEARCH_SPACE)
        logger.info("Best parameters:")
        logger.info(
            "".join([f"{d['name']}: {v:.2f} | " for d, v in zip(common.SEARCH_SPACE, best_params.squeeze())])[:-3]
        )
        
        logger.handlers[1].stream.close()
        logger.removeHandler(logger.handlers[1])
        print("_" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optim",
        help="Choose an optimization strategy.",
        type=Strategy,
        choices=list(Strategy),
        default="sobol" if SMOKE_TEST else "ei",
    )
    parser.add_argument(
        "--budget",
        help="Choose the total budget of trials (including warmup).",
        type=int,
        default=50 if SMOKE_TEST else 500,
    )
    parser.add_argument(
        "--init",
        help="Choose the number of warmup trials.",
        type=int,
        default=10 if SMOKE_TEST else 200,
    )
    parser.add_argument(
        "--runs",
        help="Specify the number of MC runs for each strategy.",
        type=int,
        default=1 if SMOKE_TEST else 30,
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
    parser.add_argument(
        "--simulate",
        help="Simulate the objective function.",
        action="store_true",
    )
    args = parser.parse_args()
    main(args)
