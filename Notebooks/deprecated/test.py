from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.measurement.synthetic_functions import branin
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
import numpy as np

N_WARMUP = 5
q = 4
N_TOTAL = N_WARMUP + 5
N_RESTARTS = 20
N_SAMPLES = 128

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
            model_kwargs={"seed": 2023},
            model_gen_kwargs={}
        ),
        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=-1,
            max_parallelism=None,
            model_kwargs={
                "surrogate": Surrogate(
                    botorch_model_class=SingleTaskGP,
                    mll_class=ExactMarginalLogLikelihood
                ),
                "botorch_acqf_class": qExpectedImprovement,
                "acquisition_options": {
                    "optimizer_options": {
                        "q": q,
                        "num_restarts": N_RESTARTS,
                        "raw_samples": N_SAMPLES
                    }
                },
            },
            # model_gen_kwargs={
            #     "model_gen_options": {
            #         "optimizer_kwargs": {
            #             "num_restarts": N_RESTARTS,
            #             "raw_samples": N_SAMPLES
            #         }
            #     }
            # }
        )
    ]
)

ax_client = AxClient(generation_strategy=gs)
ax_client.create_experiment(
    name="branin_test_experiment",
    parameters=[
        {
            "name": "x1",
            "type": "range",
            "bounds": [-5.0, 10.0],
            "value_type": "float",
            "log_scale": False
        },
        {
            "name": "x2",
            "type": "range",
            "bounds": [0.0, 15.0],
            "value_type": "float",
            "log_scale": False
        }
    ],
    objectives={"branin": ObjectiveProperties(minimize=True)}
)

for i in range(N_TOTAL):
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))