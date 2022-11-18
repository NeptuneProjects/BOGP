#!/usr/bin/env python3

from concurrent.futures import as_completed, ProcessPoolExecutor

import logging
import multiprocessing
import os
from pathlib import Path
import pickle
import random
import shutil

import numpy as np
from tqdm import tqdm

from .acoustics import MatchedFieldProcessor
from .optimization import optimizer, random_search
from tritonoa.kraken import run_kraken
from . import utils

logging.basicConfig(encoding="utf-8", level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.propagate = False
logfmt = logging.Formatter(
    fmt="%(asctime)s | %(name)-27s | %(levelname)-8s | %(message)s"
)


class Simulator:
    def __init__(self, config: dict):
        self.config = config

    def simulate(self, simulation: str, **kwargs):
        simulator = self.get_simulation(simulation)
        return simulator(**kwargs)

    def get_simulation(self, simulation: str):
        if simulation == "range" or simulation == "r":
            return self._run_range_estimation
        elif simulation == "localize" or simulation == "l":
            return self._run_localization

    def run(self, config: dict):
        config["experiment_path"] = config["path"] / config["desc"].replace(" ", "__")
        os.makedirs(config["experiment_path"], exist_ok=True)

        fh = logging.FileHandler(config["experiment_path"] / "debug.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logfmt)
        logger.addHandler(fh)
        optimizer.logger.setLevel(logging.DEBUG)
        optimizer.logger.addHandler(fh)
        optimizer.logger.propagate = False
        config["environment_config"]["tmpdir"] = config["experiment_path"] / "measured"
        logger.info("Running KRAKEN with true parameters.")
        measured_data = self.run_with_true_parameters(config["environment_config"])
        logger.info("Ran KRAKEN with true parameters.")
        config["measured_data"] = measured_data
        np.savez(
            config["environment_config"]["tmpdir"] / "measured.npz", **measured_data
        )
        with open(config["environment_config"]["tmpdir"] / "parameters.pkl", "wb") as f:
            pickle.dump(config["environment_config"], f)
        logger.info("Data saved from KRAKEN run with true parameters.")
        dispatcher(config)

    # SIMULATION: RANGE ESTIMATION
    def _run_range_estimation(self, workers: int = 1, path=None, device=None):
        NAME = "range_estimation"
        RANGE = [3.0, 6.0, 10.0, 0.5]
        # RANGE = [6.0, 10.0, 0.5]
        self.config["simulation_config"]["rec_r"] = RANGE
        config = {
            "name": NAME,
            "workers": workers,
            "path": path / NAME,
            "device": device,
        } | self.config
        self.run(config)

    # # SIMULATION: RANGE & DEPTH LOCALIZATION
    def _run_localization(self, workers: int = 1, path=None, device=None):
        logger.info("Running localization simulation.")
        # NAME = "localization"
        # RANGE = [3.0, 6.0, 10.0, 0.5]
        # DEPTH = [62]
        # self.config["simulation_config"]["rec_r"] = RANGE
        # self.config["simulation_config"]["src_z"] = DEPTH
        config = {
            # "name": NAME,
            "workers": workers,
            "path": path / self.config["name"],
            "device": device,
        } | self.config
        self.run(config)

    @staticmethod
    def run_with_true_parameters(parameters):
        p = run_kraken(parameters)
        w = p / np.linalg.norm(p)
        K = w.dot(w.conj().T)
        return {"pressure": p, "covariance": K}


def import_botorch_acquisition_function(acq_func: str):
    def _import_analytical_acq_func(acq_func: str):
        MODULE = "botorch.acquisition.analytic"
        return optimizer.import_from_str(MODULE, acq_func)

    def _import_mc_acq_func(acq_func: str):
        MODULE = "botorch.acquisition.monte_carlo"
        return optimizer.import_from_str(MODULE, acq_func)

    if acq_func[0] == "q":
        return _import_mc_acq_func(acq_func)
    else:
        return _import_analytical_acq_func(acq_func)


def import_botorch_sampler(sampler: str):
    MODULE = "botorch.sampling.samplers"
    try:
        return optimizer.import_from_str(MODULE, sampler)
    except TypeError:
        return None


def format_config_kwargs(config: dict):
    acq_func_config = config["simulation_config"]["acq_func"]
    optimizer_config = config["optimizer_config"]
    config_kwargs = {
        "acq_func": import_botorch_acquisition_function(
            acq_func_config.get("acq_func")
        ),
        "acq_func_kwargs": acq_func_config.get("acq_func_kwargs"),
        "sampler": import_botorch_sampler(acq_func_config.get("sampler")),
        "sampler_kwargs": acq_func_config.get("sampler_kwargs"),
        "n_warmup": optimizer_config.get("n_warmup"),
        "n_total": optimizer_config.get("n_total"),
        "q": 5 if acq_func_config.get("acq_func") == "qExpectedImprovement" else 1,
    }
    return config_kwargs


def format_optim_kwargs(config: dict):
    optim_kwargs = {
        "obj_func": MatchedFieldProcessor(
            config["measured_data"].get("covariance"),
            config["environment_config"],
        ),
        "search_parameters": config["optimizer_config"]["search_parameters"],
        # "device": config.get("device"),
        "seed": config.get("trial_seed"),
    }
    return optim_kwargs


def execute_bayesian_optimization(config, optim_kwargs: dict, trial_path):
    config_kwargs = format_config_kwargs(config)

    logger.info("Initializing optimizer configuration.")
    optim_config = optimizer.BayesianOptimizationGPConfig(**config_kwargs)
    logger.info("Initialized optimizer configuration.")

    logger.info("Initializing optimizer.")
    optim = optimizer.BayesianOptimizationGP(
        optim_config, device=config.get("device"), **optim_kwargs
    )
    logger.info("Initialized optimizer.")

    logger.info("Running optimization loop.")
    results = optim.run(disable_pbar=True)
    logger.info("Ran optimization loop.")

    logger.info("Saving optimization results.")
    optim.save(trial_path / "optim.pth")
    results.save(trial_path / "results.pth")
    logger.info("Saved optimization results.")


def execute_random_search(config: dict, optim_kwargs: dict, trial_path):
    logger.info("Initializing random search configuration.")
    optim_config = random_search.RandomSearchConfig(
        config["optimizer_config"].get("n_total")
    )
    logger.info("Initialized random search configuration.")

    logger.info("Initializing random search.")
    optim = optimizer.RandomSearch(optim_config, **optim_kwargs)
    logger.info("Initialized random search.")

    logger.info("Running optimization loop.")
    results = optim.run()
    logger.info("Ran optimization loop.")

    logger.info("Saving random search results.")
    results.save(trial_path / "results.pth")
    logger.info("Saved random search results.")


def worker(config: dict):
    trial_path = (
        config.get("experiment_path") / "Runs" / f"{config.get('trial_seed'):010d}"
    )

    if trial_path.exists():
        logger.info(f"Trial run directory {trial_path} exists.")
        dir_contents = [i.name for i in trial_path.glob("*")]
        if ("optim.pth" in dir_contents) and ("results.pth" in dir_contents):
            logger.info(f"optim.pth and results.pth exist.")
            logger.info(f"Skipping this run.")
            return

    config["environment_config"]["tmpdir"] = trial_path
    config["environment_config"]["title"] = f"{config.get('trial_seed'):010d}"

    logger.info(f"Creating trial run directory {trial_path}")
    os.makedirs(trial_path, exist_ok=True)
    logger.info(f"Created trial run directory {trial_path}")

    optim_kwargs = format_optim_kwargs(config)

    if config["random"]:
        execute_random_search(config, optim_kwargs, trial_path)
    elif config["SAGA"]:
        pass
    else:
        execute_bayesian_optimization(config, optim_kwargs, trial_path)

    logger.info("Cleaning up acoustic modeling files.")
    utils.clean_up_kraken_files(trial_path)
    logger.info("Cleaned up acoustic modeling files.")


def dispatcher(config: dict):
    n_trials = config.get("n_sim")
    random.seed(config.get("main_seed"))
    logger.info(
        f"Running {n_trials} evaluations using master random seed {config.get('main_seed')}."
    )

    pbar_kwargs = {
        "desc": "Running MC",
        "bar_format": "{l_bar}{bar:20}{r_bar}{bar:-20b}",
        "leave": True,
        "position": 0,
        "unit": "eval",
        "disable": True,
        "colour": "blue",
        "total": n_trials,
    }
    # if config.get("workers") == 1:
    for _ in range(n_trials):
        worker(config | {"trial_seed": random.randint(0, int(1e9))})
    # else:
    #     with ProcessPoolExecutor(
    #         max_workers=config.get("workers"),
    #         mp_context=multiprocessing.get_context("spawn"),
    #     ) as executor:
    #         futures = [
    #             executor.submit(
    #                 worker, config | {"trial_seed": random.randint(0, int(1e9))}
    #             )
    #             for _ in range(n_trials)
    #         ]

    #         results = []
    #         for future in tqdm(as_completed(futures), **pbar_kwargs):
    #             results.append(future.result())

    configpath = Path(config["configpath"])
    logger.info("Moving configuration file from queue to run directory.")
    shutil.move(
        configpath.absolute(), (config["experiment_path"] / configpath.name).absolute()
    )
    logger.info("Moved configuration file from queue to run directory.")


# class Worker:
#     def __init__(self, config):
#         self.config = config

#     def execute(self, **kwargs):
#         executor = self.get_executor()
#         return executor(**kwargs)

#     def get_executor(self):
#         if self.config["random"]:
#             return self._execute_random_search()
#         elif self.config["SAGA"]:
#             return self._execute_SAGA()
#         else:
#             return self._execute_BOGP()

#     def _execute_random_search(self):
#         return

#     def _execute_SAGA(self):
#         return

#     def _execute_BOGP(self):
#         return
