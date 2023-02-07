#!/usr/bin/env python3

from argparse import ArgumentParser
from dataclasses import dataclass, field
from itertools import product
import os
from pathlib import Path
import random
import tomli

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.service.ax_client import ObjectiveProperties
from botorch.acquisition import qExpectedImprovement, qProbabilityOfImprovement
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
import numpy as np

from acoustics import covariance
from oao.common import UNINFORMED_STRATEGIES
from oao.utilities import save_config
import swellex
from tritonoa.kraken import run_kraken
from tritonoa.sp import added_wng, snrdb_to_sigma


SIM_SCENARIOS = {"rec_r": [1.0, 3.0, 5.0, 7.0], "src_z": [62.0], "snr": [20]}
SKIP_TIMESTEPS = (
    list(range(73, 85))
    + list(range(95, 103))
    + list(range(187, 199))
    + list(range(287, 294))
    + list(range(302, 309))
)
EXP_SCENARIOS = {"timestep": [i for i in range(350) if i not in SKIP_TIMESTEPS]}
RANGE_ESTIMATION_PARAMETERS = [
    {
        "name": "rec_r",
        "type": "range",
        "bounds": [0.01, 10.0],
        "value_type": "float",
        "log_scale": False,
    },
]
LOCALIZATION_PARAMETERS = [
    {
        "name": "rec_r",
        "type": "range",
        "bounds": [0.01, 10.0],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "src_z",
        "type": "range",
        "bounds": [1.0, 200.0],
        "value_type": "float",
        "log_scale": False,
    },
]
FREQUENCIES = swellex.HIGH_SIGNAL_TONALS[6:]
EXP_DATADIR = (Path.cwd() / "Data" / "SWELLEX96" / "VLA" / "selected").relative_to(
    Path.cwd()
)


@dataclass(kw_only=True)
class SimulationConfig:
    num_mc_runs: int = 1
    num_restarts: int = 40
    num_samples: int = 1024
    num_trials: int = 200
    num_warmup: int = 10
    q: int = 5
    root: str = "Data"
    main_seed: int = 2009
    serial: str = "serial"
    evaluation_config = None
    frequencies: list = field(default_factory=list)

    def __post_init__(self):
        self.root = Path(self.root)
        self.scenarios = SIM_SCENARIOS
        self.strategies = [
            {
                "loop_type": "grid",
                "num_trials": self.num_trials,
            },
            {
                "loop_type": "lhs",
                "num_trials": self.num_trials,
            },
            {
                "loop_type": "random",
                "num_trials": self.num_trials,
            },
            {
                "loop_type": "sobol",
                "num_trials": self.num_trials,
            },
            {
                "loop_type": "sequential",
                "num_trials": self.num_trials,
                "batch_size": 1,
                "generation_strategy": GenerationStrategy(
                    [
                        GenerationStep(
                            model=Models.SOBOL,
                            num_trials=self.num_warmup,
                            max_parallelism=self.num_warmup,
                            # model_kwargs={"seed": self.main_seed},
                        ),
                        GenerationStep(
                            model=Models.BOTORCH_MODULAR,
                            num_trials=self.num_trials - self.num_warmup,
                            max_parallelism=None,
                            model_kwargs={
                                "surrogate": Surrogate(
                                    botorch_model_class=SingleTaskGP,
                                    mll_class=ExactMarginalLogLikelihood,
                                ),
                                "botorch_acqf_class": qExpectedImprovement,
                            },
                            model_gen_kwargs={
                                "model_gen_options": {
                                    "optimizer_kwargs": {
                                        "num_restarts": self.num_restarts,
                                        "raw_samples": self.num_samples,
                                    }
                                }
                            },
                        ),
                    ]
                ),
            },
            {
                "loop_type": "sequential",
                "num_trials": self.num_trials,
                "batch_size": 1,
                "generation_strategy": GenerationStrategy(
                    [
                        GenerationStep(
                            model=Models.SOBOL,
                            num_trials=self.num_warmup,
                            max_parallelism=self.num_warmup,
                            # model_kwargs={"seed": self.main_seed},
                        ),
                        GenerationStep(
                            model=Models.BOTORCH_MODULAR,
                            num_trials=self.num_trials - self.num_warmup,
                            max_parallelism=None,
                            model_kwargs={
                                "surrogate": Surrogate(
                                    botorch_model_class=SingleTaskGP,
                                    mll_class=ExactMarginalLogLikelihood,
                                ),
                                "botorch_acqf_class": qProbabilityOfImprovement,
                            },
                            model_gen_kwargs={
                                "model_gen_options": {
                                    "optimizer_kwargs": {
                                        "num_restarts": self.num_restarts,
                                        "raw_samples": self.num_samples,
                                    }
                                }
                            },
                        ),
                    ]
                ),
            },
            {
                "loop_type": "greedy_batch",
                "num_trials": self.num_trials,
                "batch_size": self.q,
                "generation_strategy": GenerationStrategy(
                    [
                        GenerationStep(
                            model=Models.SOBOL,
                            num_trials=self.num_warmup,
                            max_parallelism=self.num_warmup,
                            # model_kwargs={"seed": self.main_seed},
                        ),
                        GenerationStep(
                            model=Models.BOTORCH_MODULAR,
                            num_trials=self.num_trials - self.num_warmup,
                            max_parallelism=self.q,
                            model_kwargs={
                                "surrogate": Surrogate(
                                    botorch_model_class=SingleTaskGP,
                                    mll_class=ExactMarginalLogLikelihood,
                                ),
                                "botorch_acqf_class": qExpectedImprovement,
                            },
                            model_gen_kwargs={
                                "model_gen_options": {
                                    "optimizer_kwargs": {
                                        "num_restarts": self.num_restarts,
                                        "raw_samples": self.num_samples,
                                    }
                                }
                            },
                        ),
                    ]
                ),
            },
        ]
        self.experiment_kwargs = {
            "name": "test_mfp",
            "objectives": {"bartlett": ObjectiveProperties(minimize=False)},
        }
        self.obj_func_parameters = {
            "env_parameters": swellex.environment,
            "frequencies": self.frequencies,
        }


@dataclass(kw_only=True)
class RangeEstimationSimulationConfig(SimulationConfig):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_kwargs["parameters"] = RANGE_ESTIMATION_PARAMETERS


@dataclass(kw_only=True)
class LocalizationSimulationConfig(SimulationConfig):
    num_grid_trials: int = 10

    def __post_init__(self):
        super().__post_init__()
        i = next(
            i for i, item in enumerate(self.strategies) if item["loop_type"] == "grid"
        )
        self.strategies[i]["num_trials"] = self.num_grid_trials
        self.experiment_kwargs["parameters"] = LOCALIZATION_PARAMETERS


@dataclass(kw_only=True)
class ExperimentalConfig(SimulationConfig):
    def __post_init__(self):
        super().__post_init__()
        self.datadir = EXP_DATADIR
        self.scenarios = EXP_SCENARIOS


@dataclass(kw_only=True)
class RangeEstimationExperimentalConfig(ExperimentalConfig):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_kwargs["parameters"] = RANGE_ESTIMATION_PARAMETERS


@dataclass(kw_only=True)
class LocalizationExperimentalConfig(ExperimentalConfig):
    num_grid_trials: int = 10

    def __post_init__(self):
        super().__post_init__()
        super().__post_init__()
        i = next(
            i for i, item in enumerate(self.strategies) if item["loop_type"] == "grid"
        )
        self.strategies[i]["num_trials"] = self.num_grid_trials
        self.experiment_kwargs["parameters"] = LOCALIZATION_PARAMETERS


class Initializer:
    def __init__(self, Config):
        self.Config = Config
        self.optim = self.get_optimization_problem(self.Config)
        self.mode = self.get_optimization_mode(self.Config)
        self.seeds = self.get_mc_seeds(self.Config.main_seed, self.Config.num_mc_runs)

    def configure(self):
        self.make_folders()
        scenarios = self.get_scenario_dict(**self.Config.scenarios)

        if self.mode == "experimental":
            p = []
            for f in self.Config.frequencies:
                p.append(np.load(self.Config.datadir / f"{f:.1f}Hz" / "data.npy"))
            p = np.array(p)

        for j, scenario in enumerate(scenarios):
            scenario_folder, data_folder = self.make_scenario_folder(scenario)

            K = []
            for i, f in enumerate(self.Config.frequencies):

                if self.mode == "simulation":
                    K.append(
                        self.simulate_measurement_covariance(
                            self.Config.obj_func_parameters["env_parameters"]
                            | scenario
                            | {"freq": f, "tmpdir": data_folder, "title": f"{f:.1f}Hz"}
                        )
                    )
                elif self.mode == "experimental":
                    K.append(covariance(p[i, j, ...]))

            K = np.array(K)
            if len(K.shape) == 2:
                K = K[np.newaxis, ...]
            np.save(data_folder / "measurement_covariance.npy", K)

            param_names = [
                d["name"] for d in self.Config.experiment_kwargs["parameters"]
            ]

            for k in scenario.keys():
                if k not in param_names:
                    self.Config.obj_func_parameters["env_parameters"][k] = scenario[k]

            for strategy in self.Config.strategies:
                if strategy["loop_type"] not in UNINFORMED_STRATEGIES:
                    if (
                        strategy["generation_strategy"]
                        ._steps[1]
                        .model_kwargs["botorch_acqf_class"]
                        .__name__
                        == "qExpectedImprovement"
                    ):
                        acqf = "_qei"
                    elif (
                        strategy["generation_strategy"]
                        ._steps[1]
                        .model_kwargs["botorch_acqf_class"]
                        .__name__
                        == "qProbabilityOfImprovement"
                    ):
                        acqf = "_qpi"
                    elif (
                        strategy["generation_strategy"]
                        ._steps[1]
                        .model_kwargs["botorch_acqf_class"]
                        .__name__
                        == "qUpperConfidenceBound"
                    ):
                        acqf = "_qucb"
                else:
                    acqf = ""

                for seed in self.seeds:
                    self.Config.experiment_kwargs["name"] = (
                        f"{self.optim};"
                        f"{self.mode};{self.Config.serial};{scenario};"
                        f"{strategy['loop_type']}{acqf}"
                        f"{(str(strategy['batch_size']) if 'batch_size' in strategy.keys() else '')};"
                        f"{seed}"
                    )
                    run_config = {
                        "experiment_kwargs": self.Config.experiment_kwargs,
                        "seed": seed,
                        "strategy": strategy,
                        "obj_func_parameters": self.Config.obj_func_parameters,
                        "evaluation_config": self.Config.evaluation_config,
                        "destination": str(
                            scenario_folder.relative_to(self.mode_folder)
                        ),
                    }

                    config_name = (
                        f"config__{scenario_folder.name}__"
                        f"{strategy['loop_type'] + acqf}__{seed:010d}.json"
                    )
                    save_config(self.q_folder / config_name, run_config)

    @staticmethod
    def get_mc_seeds(main_seed: int, num_runs: int) -> list:
        random.seed(main_seed)
        return [random.randint(0, int(1e9)) for _ in range(num_runs)]

    @staticmethod
    def get_optimization_mode(Config):
        if isinstance(Config, ExperimentalConfig):
            return "experimental"
        elif isinstance(Config, SimulationConfig):
            return "simulation"

    @staticmethod
    def get_optimization_problem(Config):
        # The order of checking instance is important since
        # ExperimentConfig inherits from SimulationConfig
        if isinstance(Config, RangeEstimationSimulationConfig) or isinstance(
            Config, RangeEstimationExperimentalConfig
        ):
            return "range_estimation"
        elif isinstance(Config, LocalizationSimulationConfig) or isinstance(
            Config, LocalizationExperimentalConfig
        ):
            return "localization"

    @staticmethod
    def get_scenario_dict(**kwargs):
        keys = kwargs.keys()
        vals = kwargs.values()
        for i in product(*vals):
            yield dict(zip(keys, i))

    @staticmethod
    def get_scenario_path(scenario: dict) -> str:
        return "__".join([f"{k}={v}" for k, v in scenario.items()])

    def make_folders(self):
        self.mode_folder = (
            self.Config.root / self.optim / self.mode / self.Config.serial
        )
        self.q_folder = self.mode_folder / "queue"
        os.makedirs(self.mode_folder)
        os.mkdir(self.q_folder)

    def make_scenario_folder(self, scenario):
        scenario_name = self.get_scenario_path(scenario)
        scenario_folder = self.mode_folder / scenario_name
        data_folder = scenario_folder / "data"
        os.makedirs(data_folder)
        return scenario_folder, data_folder

    @staticmethod
    def simulate_measurement_covariance(env_parameters: dict) -> np.array:
        snr = env_parameters.pop("snr")
        sigma = snrdb_to_sigma(snr)
        p = run_kraken(env_parameters)
        p /= np.linalg.norm(p)
        noise = added_wng(p.shape, sigma=sigma, cmplx=True)
        p += noise
        K = p.dot(p.conj().T)
        return K


def load_toml(path) -> dict:
    if isinstance(path, str):
        path = Path(path)
    with open(path, "rb") as fp:
        return tomli.load(fp)


def main(path, optim, modes):
    config = load_toml(path)
    optimizations = []
    if "r" in optim:
        if "s" in modes:
            optimizations.append(
                RangeEstimationSimulationConfig(
                    **config["range_estimation"]["simulation"]
                )
            )
        if "e" in modes:
            optimizations.append(
                RangeEstimationExperimentalConfig(
                    **config["range_estimation"]["experimental"]
                )
            )
    if "l" in optim:
        if "s" in modes:
            optimizations.append(
                LocalizationSimulationConfig(**config["localization"]["simulation"])
            )
        if "e" in modes:
            optimizations.append(
                LocalizationExperimentalConfig(**config["localization"]["experimental"])
            )

    for opt in optimizations:
        Init = Initializer(opt)
        Init.configure()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=str, help="Path to config.toml")
    parser.add_argument("optim", type=str, help="Type of optimization problem")
    parser.add_argument("modes", type=str, help="Simulation or experimental")
    args = parser.parse_args()
    main(args.path, args.optim, args.modes)
