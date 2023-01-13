import sys
sys.settrace

from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import render, init_notebook_plotting

from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models.gp_regression import FixedNoiseGP, SingleTaskGP
from botorch.models.model import Model

from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

import numpy as np

import torch

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_WARMUP = 3
N_TOTAL = 20
N_RESTARTS = 20
N_SAMPLES = 512
q = 5

def evaluate(parameters):
    x = np.array([parameters.get(f"x{i+1}") for i in range(6)])
    # In our case, standard error is 0, since we are computing a synthetic function.
    # return {"hartmann6": (hartmann6(x), 0.0), "l2norm": (np.sqrt((x ** 2).sum()), 0.0)}
    return {"hartmann6": hartmann6(x), "l2norm": np.sqrt((x ** 2).sum())}

gs = GenerationStrategy(
    steps=[
        GenerationStep(
            model=Models.SOBOL,
            num_trials=N_WARMUP,
            min_trials_observed=3,
            max_parallelism=N_WARMUP,
            model_kwargs={"seed": 2009},
            model_gen_kwargs={}
        ),
        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=-1,
            max_parallelism=None,
            model_kwargs={
                # "torch_device": device,
                "surrogate": Surrogate(
                    botorch_model_class=SingleTaskGP,
                    mll_class=ExactMarginalLogLikelihood
                ),
                "botorch_acqf_class": qExpectedImprovement,
                "acquisition_options": {
                    "num_restarts": N_RESTARTS,
                    "raw_samples": N_SAMPLES,
                    "q": q
                }
            },
            should_deduplicate=True
        )
    ]
)

ax_client = AxClient(generation_strategy=gs, verbose_logging=True)

ax_client.create_experiment(
    name="hartmann_test_experiment",
    parameters=[
        {
            "name": "x1",
            "type": "range",
            "bounds": [0.0, 1.0],
            "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            "log_scale": False,  # Optional, defaults to False.
        },
        {
            "name": "x2",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
        {
            "name": "x3",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
        {
            "name": "x4",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
        {
            "name": "x5",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
        {
            "name": "x6",
            "type": "range",
            "bounds": [0.0, 1.0],
        },
    ],
    objectives={"hartmann6": ObjectiveProperties(minimize=True)},
    parameter_constraints=["x1 + x2 <= 2.0"],  # Optional.
    outcome_constraints=["l2norm <= 1.25"],  # Optional.
)

for i in range(N_TOTAL):
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))
