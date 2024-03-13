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
import warnings

import numpy as np

import baxus, ei, gibbon, helpers, obj, pi, sobol, ucb, grid, common, rand

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
    GRID = "grid"
    RANDOM = "random"

    def __str__(self):
        return self.value


def get_loop(
    optim: Strategy, dummy_dim: int, budget: int, n_init: int, beta: float
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
        kwargs = ucb.UCBLoopArgs(dim=TRUE_DIM, budget=budget, n_init=n_init, beta=beta)
    if optim == Strategy.GRID:
        loop = grid.loop
        kwargs = grid.GridLoopArgs(dim=TRUE_DIM, budget=budget)
    if optim == Strategy.RANDOM:
        loop = rand.loop
        kwargs = rand.RandomLoopArgs(dim=TRUE_DIM, budget=budget)
    return loop, asdict(kwargs)


def main(args) -> None:
    Path.mkdir(args.dir / args.serial, parents=True, exist_ok=True)

    loop, kwargs = get_loop(
        args.optim, dummy_dim=args.ndummy, budget=args.budget, n_init=args.init, beta=args.beta
    )
    dtype = kwargs.pop("dtype", None)
    device = kwargs.pop("device", None)

    random.seed(args.seed)
    seeds = helpers.get_random_seeds(args.runs)

    print("=" * 100)
    for seed in seeds:
        fname = f"{args.optim}_{args.budget}-{args.init}_{seed:04d}"
        helpers.initialize_logger_file(args.dir / args.serial / f"{fname}.log", logger, logfmt)

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

        with open(args.dir / args.serial / f"{fname}.json", "w") as f:
            json.dump(kwargs, f, indent=4)

        np.savez(
            args.dir / args.serial / fname,
            X=X,
            Y=Y,
            t=times,
        )

        print("-" * 100)
        logger.info("*** Optimization complete. ***")
        helpers.log_best_value_and_parameters(X, Y, common.SEARCH_SPACE)
        logger.handlers[1].stream.close()
        logger.removeHandler(logger.handlers[1])
        print("=" * 100)

        if args.optim == Strategy.GRID:
            break


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
        "--serial",
        help="Specify the serial name (appended to --dir).",
        type=str,
        default="",
    )
    parser.add_argument(
        "--ndummy",
        help="[BAxUS only] Specify number of dummy dimensions.",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--beta",
        help="[UCB only] Specify beta for exploration/exploitation.",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--simulate",
        help="Simulate the objective function.",
        action="store_true",
    )
    args = parser.parse_args()
    main(args)
