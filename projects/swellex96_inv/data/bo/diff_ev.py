#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
from functools import partial
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, differential_evolution
from tqdm import tqdm
from tritonoa.sp.mfp import MatchedFieldProcessor

import common
from obj import get_objective

parameter_keys = [
    "rec_r",
    "src_z",
    "tilt",
    "h_w",
    "h_sed",
    "c_p_sed_top",
    "dc_p_sed",
    # "a_p_sed",
    # "rho_sed",
]

full_bounds = [
    (1.05, 1.09),
    (69.0, 73.0),
    (1.8, 2.3),
    (216.0, 222.0),
    (20.0, 26.0),
    (1565.0, 1575.0),
    (18.0, 23.0),
    (1588.0, 1598.0),
    (0.0, 1.0),
    (1.75, 2.25),
]


class Callback:
    def __init__(self):
        self.start_time = time.time()
        self.elapsed_time = []
        self.fun = []
        self.nit = []
        self.nfev = []
        self.x = []
        self.population = []
        self.population_energies = []

    def __call__(self, intermediate_result: OptimizeResult) -> None:
        self.elapsed_time.append(time.time() - self.start_time)
        self.fun.append(intermediate_result.fun)
        self.nit.append(intermediate_result.nit)
        self.nfev.append(intermediate_result.nfev)
        self.x.append(intermediate_result.x)
        self.population.append(intermediate_result.population)
        self.population_energies.append(intermediate_result.population_energies)
        # vals = [f"{key}={self.x[-1][i]}" for i, key in enumerate(parameter_keys)]
        # print(
        #     f"Time: {self.elapsed_time[-1]:.2f} s, "
        #     f"Obj: {self.fun[-1]}, "
        #     f"It: {self.nit[-1]}, "
        #     f"nfev: {self.nfev[-1]} | "
        #     f"{' | '.join(vals)}"
        # )

    def save_results(self, path: Path = Path("de_results.npz")) -> None:
        np.savez(
            path,
            elapsed_time=self.elapsed_time,
            fun=self.fun,
            nit=self.nit,
            nfev=self.nfev,
            x=self.x,
            population=self.population,
            population_energies=self.population_energies,
        )

    @staticmethod
    def write_csv(data: tuple) -> None:
        with open("output.csv", "a") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow(data)


def fitness_func(x: np.ndarray, objective: MatchedFieldProcessor = None) -> float:
    return objective({k: v for k, v in zip(parameter_keys, x)})


def main(args: argparse.Namespace) -> None:

    if args.full:
        print("Running full DE.")
        num_runs = 1
        bounds = full_bounds
        de_kwargs = {
            "maxiter": 100,
            "popsize": 100,
            "mutation": (0.7, 1.0),
            "recombination": 0.1,
            "polish": True,
            "updating": "immediate",
        }
        savename = "de_results_full"
    else:
        print("Running vanilla DE.")
        num_runs = args.num_runs
        bounds = [(i["bounds"][0], i["bounds"][1]) for i in common.SEARCH_SPACE]
        parameter_keys = [i["name"] for i in common.SEARCH_SPACE]
        de_kwargs = {
            "maxiter": 100,
            "popsize": 10,
            "mutation": 0.5,
            "recombination": 0.1,
            "polish": False,
            "updating": "deferred",
        }
        savename = "de_results"

    objective = get_objective(simulate=False)
    savepath = common.SWELLEX96Paths.outputs / "runs" / "de"
    savepath.mkdir(parents=True, exist_ok=True)

    alldata = []
    for i in tqdm(
        range(num_runs), total=num_runs, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"
    ):
        callback = Callback()
        differential_evolution(
            partial(fitness_func, objective=objective),
            bounds,
            seed=i if not args.full else 719,
            disp=True,
            callback=callback,
            init="latinhypercube",
            **de_kwargs,
        )
        callback.save_results(path=savepath / f"de_results{i:02d}.npz")
        data = pd.DataFrame(data=callback.x, columns=parameter_keys)
        data["seed"] = i
        data["wall_time"] = callback.elapsed_time
        data["nit"] = callback.nit
        data["nfev"] = callback.nfev
        data["obj"] = callback.fun
        alldata.append(data)

    df = pd.concat(alldata, ignore_index=True)
    df.to_csv(savepath / f"{savename}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_runs", type=int, default=100)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    main(args)
