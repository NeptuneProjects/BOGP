#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

NUM_RUNS = 30

parameter_keys = [i["name"] for i in common.SEARCH_SPACE]


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


def main() -> None:
    objective = get_objective(simulate=False)
    bounds = [(i["bounds"][0], i["bounds"][1]) for i in common.SEARCH_SPACE]
    savepath = common.SWELLEX96Paths.outputs / "runs" / "de"
    savepath.mkdir(parents=True, exist_ok=True)

    alldata = []
    for i in tqdm(
        range(NUM_RUNS), total=NUM_RUNS, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"
    ):
        callback = Callback()
        differential_evolution(
            partial(fitness_func, objective=objective),
            bounds,
            maxiter=300,
            popsize=10,
            mutation=0.5, # Also known as differential weight, or "F"
            recombination=0.1, # Also known as crossover probability, or "CR"
            seed=i,
            disp=True,
            callback=callback,
            polish=False,
            init="latinhypercube",
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
    df.to_csv(savepath / "de_results.csv", index=False)


if __name__ == "__main__":
    main()
