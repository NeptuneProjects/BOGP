#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path
import tomli

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.service.ax_client import ObjectiveProperties
from botorch.acquisition import qExpectedImprovement, qProbabilityOfImprovement
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

import swellex

SIM_SCENARIOS = {"rec_r": [1.0, 3.0, 5.0, 7.0], "src_z": [62.0], "snr": [20]}
EXP_SCENARIOS = {"timestep": range(350)}
RANGE_ESTIMATION_PARAMETERS = [
    {
        "name": "rec_r",
        "type": "range",
        "bounds": [0.1, 10.0],
        "value_type": "float",
        "log_scale": False,
    },
]
LOCALIZATION_PARAMETERS = [
    {
        "name": "rec_r",
        "type": "range",
        "bounds": [0.1, 10.0],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "src_z",
        "type": "range",
        "bounds": [0.0, 200.0],
        "value_type": "float",
        "log_scale": False,
    },
]
EXP_DATADIR = (Path.cwd() / "Data" / "SWELLEX96" / "VLA" / "selected" / "data.npy").relative_to(Path.cwd())

@dataclass
class Simulation:
    num_mc_runs: int = 1
    num_restarts: int = 40
    num_samples: int = 1024
    num_trials: int = 200
    num_warmup: int = 10
    q: int = 1
    root: str = "Data"
    main_seed: int = 2009
    serial: str = "serial"
    evaluation_config = None

    def __post_init__(self):
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
                "loop_type": "sequential",  # <--- This is where to specify loop,
                "num_trials": self.num_trials,
                "batch_size": 1,
                "generation_strategy": GenerationStrategy(
                    [
                        GenerationStep(
                            model=Models.SOBOL,
                            num_trials=self.num_warmup,
                            max_parallelism=self.num_warmup,
                            model_kwargs={
                                "seed": self.main_seed
                            },  # TODO: Fix seed handling
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
                "loop_type": "sequential",  # <--- This is where to specify loop,
                "num_trials": self.num_trials,
                "batch_size": 1,
                "generation_strategy": GenerationStrategy(
                    [
                        GenerationStep(
                            model=Models.SOBOL,
                            num_trials=self.num_warmup,
                            max_parallelism=self.num_warmup,
                            model_kwargs={"seed": self.main_seed},
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
                "loop_type": "greedy_batch",  # <--- This is where to specify loop,
                "num_trials": self.num_trials,
                "batch_size": self.q,
                "generation_strategy": GenerationStrategy(
                    [
                        GenerationStep(
                            model=Models.SOBOL,
                            num_trials=self.num_warmup,
                            max_parallelism=self.num_warmup,
                            model_kwargs={"seed": self.main_seed},
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
        self.obj_func_parameters = {"env_parameters": swellex.environment}


@dataclass(kw_only=True)
class RangeEstimationSimulation(Simulation):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_kwargs["parameters"] = RANGE_ESTIMATION_PARAMETERS


@dataclass(kw_only=True)
class LocalizationSimulation(Simulation):
    num_grid_trials: int = 10

    def __post_init__(self):
        super().__post_init__()
        i = next(
            i for i, item in enumerate(self.strategies) if item["loop_type"] == "grid"
        )
        self.strategies[i]["num_trials"] = self.num_grid_trials
        self.experiment_kwargs["parameters"] = LOCALIZATION_PARAMETERS


@dataclass(kw_only=True)
class Experimental(Simulation):
    
    def __post_init__(self):
        super().__post_init__()
        self.datadir = EXP_DATADIR
        self.scenarios = EXP_SCENARIOS
        

@dataclass(kw_only=True)
class RangeEstimationExperimental(Experimental):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_kwargs["parameters"] = RANGE_ESTIMATION_PARAMETERS


@dataclass(kw_only=True)
class LocalizationExperimental(Experimental):
    num_grid_trials: int = 10

    def __post_init__(self):
        super().__post_init__()
        super().__post_init__()
        i = next(
            i for i, item in enumerate(self.strategies) if item["loop_type"] == "grid"
        )
        self.strategies[i]["num_trials"] = self.num_grid_trials
        self.experiment_kwargs["parameters"] = LOCALIZATION_PARAMETERS
    

if __name__ == "__main__":
    with open(
        "/Users/williamjenkins/Research/Projects/BOGP/Source/scripts/config.toml", "rb"
    ) as fp:
        config = tomli.load(fp)


    optim = "rl"
    mode = "se"

    optimizations = []
    if "r" in optim:
        if "s" in mode:
            optimizations.append(
                RangeEstimationSimulation(**config["range_estimation"]["simulation"])
            )
        if "e" in mode:
            optimizations.append(
                RangeEstimationExperimental(**config["range_estimation"]["experimental"])
            )
    if "l" in optim:
        if "s" in mode:
            optimizations.append(
                LocalizationSimulation(**config["localization"]["simulation"])
            )
        if "e" in mode:
            optimizations.append(
                LocalizationExperimental(**config["localization"]["experimental"])
            )
