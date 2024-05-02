#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
from functools import partial
from pathlib import Path
import time

import numpy as np
from scipy.optimize import OptimizeResult, differential_evolution

from tritonoa.at.models.kraken import runner
from tritonoa.sp.beamforming import covariance
from tritonoa.sp.mfp import MatchedFieldProcessor
from tritonoa.sp.beamforming import beamformer

import common
from obj import get_objective

parameter_keys = [i["name"] for i in common.SEARCH_SPACE]


def fitness_func(x: np.ndarray, objective: MatchedFieldProcessor = None) -> float:
    return objective({k: v for k, v in zip(parameter_keys, x)})


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
        print(intermediate_result)

    def save_results(self, path: Path = Path("de_results.npz")) -> None:
        # with open(path, "wb") as f:
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


def main() -> None:
    objective = get_objective(simulate=False)
    bounds = [(i["bounds"][0], i["bounds"][1]) for i in common.SEARCH_SPACE]

    callback = Callback()
    result = differential_evolution(
        partial(fitness_func, objective=objective),
        bounds,
        disp=True,
        polish=False,
        maxiter=300,
        init="sobol",
        popsize=70,
        mutation=0.9,
        callback=callback,
    )
    callback.save_results(path=common.SWELLEX96Paths.outputs / "de_results.npz")
    print(result)


if __name__ == "__main__":
    main()
