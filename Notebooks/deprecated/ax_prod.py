from ax.models.torch.botorch_modular.acquisition import Acquisition
from ax.models.torch.botorch_modular.model import BoTorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate


from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.measurement.synthetic_functions import branin
from ax.utils.notebook.plotting import render, init_notebook_plotting

from botorch.acquisition.analytic import ProbabilityOfImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP
# from botorch.models.model import Model
from botorch.optim import optimize_acqf
from ax.core.observation import ObservationFeatures

from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

import numpy as np

import torch


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_WARMUP = 2 ** 4
q = 4
N_TOTAL = N_WARMUP + q
N_RESTARTS = 10
N_SAMPLES = 256


def evaluate(parameters):
    x = np.array([parameters.get(f"x{i+1}") for i in range(2)])
    return {"branin": (branin(x), None)}


gs = GenerationStrategy(
    steps=[
        GenerationStep(
            model=Models.SOBOL,
            num_trials=N_WARMUP,
            min_trials_observed=N_WARMUP,
            max_parallelism=N_WARMUP,
            model_kwargs={"seed": 2009},
            model_gen_kwargs={}
        ),
        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            # model=Models.GPEI,
            num_trials=N_TOTAL - N_WARMUP,
            max_parallelism=q,
            model_kwargs={
                # "torch_device": device,
                "surrogate": Surrogate(
                    botorch_model_class=SingleTaskGP,
                    mll_class=ExactMarginalLogLikelihood
                ),
                # "botorch_acqf_class": qExpectedImprovement,
                "botorch_acqf_class": ProbabilityOfImprovement,
                "acquisition_options": {
                    "optimizer_options": {
                        # "q": q,
                        "num_restarts": N_RESTARTS,
                        "raw_samples": N_SAMPLES
                    }
                }
            }
        )
    ]
)

print(type(Models.BOTORCH_MODULAR))

ax_client = AxClient(generation_strategy=gs, verbose_logging=True, enforce_sequential_optimization=True)

ax_client.create_experiment(
    name="branin_test_experiment",
    parameters=[
        {
            "name": "x1",
            "type": "range",
            "bounds": [-5.0, 10.0],
            "value_type": "float",  # Optional, defaults to inference from type of "bounds".
            "log_scale": False,  # Optional, defaults to False.
        },
        {
            "name": "x2",
            "type": "range",
            "bounds": [0.0, 15.0],
            "value_type": "float"
        }
    ],
    objectives={"branin": ObjectiveProperties(minimize=True)}
)

for i in range(N_TOTAL):
    # trials_to_evaluate, optimization_complete = ax_client.get_next_trials(max_trials=q)
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))
    

    test_features = [
        ObservationFeatures(parameters={"x1": 0, "x2": 0}),
        ObservationFeatures(parameters={"x1": 2, "x2": 2})
    ]
    
    if ax_client.generation_strategy.current_step.index == 1:
        model = ax_client.generation_strategy.model
        y_test = model.predict(test_features)
        alpha = model.evaluate_acquisition_function(test_features)
        print(f"y_test = {y_test}")
        print(f"alpha = {alpha}")
    # ax_client.get_next_trials()
    # trials_to_evaluate = {}
    # for _ in range(q):
    #     parameters, trial_index = ax_client.get_next_trial()
    #     trials_to_evaluate[trial_index] = parameters

    # [
    #     ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))
    #     for trial_index, parameters in trials_to_evaluate.items()
    # ]

    # results = {
    #     trial_index: evaluate(parameters) for trial_index, parameters in trials_to_evaluate
    # }

# print(ax_client.generation_strategy.trials_as_df)

