#!/usr/bin/env python3

from datetime import datetime
import logging

from ax.service.ax_client import AxClient
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.utils.instantiation import ObjectiveProperties
from ax.utils.notebook.plotting import render
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement
from gpytorch.mlls import ExactMarginalLogLikelihood
import mlflow
import numpy as np
import plotly.graph_objects as go

from acoustics import simulate_measurement_covariance
from gpmodel import ConstrainedLocalizationGP, LocalizationGP
from objective import objective_function
import swellex

client = mlflow.MlflowClient()
experiment_name = "BOGP"
try:
    experiment_id = client.create_experiment(experiment_name)
except mlflow.exceptions.MlflowException:
    current_experiment = dict(mlflow.get_experiment_by_name(experiment_name))
    experiment_id = current_experiment["experiment_id"]
    client.get_experiment(experiment_id)

logger = logging.getLogger("mlflow")
logger.setLevel(logging.DEBUG)

true_parameters = {
    "rec_r": 3.0,
    "src_z": 60.0
}
env_parameters = swellex.environment | {"tmpdir": "."}
frequencies = [148, 166, 201, 235, 283, 338, 388]
K = []
for f in frequencies:
    K.append(
        simulate_measurement_covariance(
            env_parameters | {"snr": 20, "freq": f} | true_parameters
        )
    )
K = np.array(K)
if len(K.shape) == 2:
    K = K[np.newaxis, ...]


n_warmup = 32
n_total = n_warmup + 10

gs = GenerationStrategy(
    steps=[
        GenerationStep(
            model=Models.SOBOL,
            num_trials=n_warmup,
        ),
        GenerationStep(
            model=Models.BOTORCH_MODULAR,
            num_trials=-1,
            model_kwargs={
                "surrogate": Surrogate(
                    botorch_model_class=ConstrainedLocalizationGP,
                    mll_class=ExactMarginalLogLikelihood
                ),
                "botorch_acqf_class": ExpectedImprovement,
            }
        )
    ]
)



ax_client = AxClient(generation_strategy=gs)

ax_client.create_experiment(
    name="test",
    parameters=[
        {
            "name": "rec_r",
            "type": "range",
            "bounds": [2.0, 4.0],
            "value_type": "float",
            "log_scale": False,
        },
        {
            "name": "src_z",
            "type": "range",
            "bounds": [1, 100],
            "value_type": "float",
            "log_scale": False,
        }
    ],
    objectives={
        "bartlett": ObjectiveProperties(minimize=False)
    }
)


with mlflow.start_run(experiment_id=experiment_id, run_name=datetime.now().strftime("%Y-%m-%d %H:%M:%S")) as main_run:
    
    for i in range(n_total):

        candidates, trial_index = ax_client.get_next_trial()
        parameters = {"env_parameters": env_parameters, "K": K, "frequencies": frequencies} | candidates
        raw_data = objective_function(parameters)

        # Log metrics
        # logger.report_scalar(title="Range Error [km]", series="rec_r", iteration=trial_index, value=candidates["rec_r"])
        # logger.report_scalar(title="Depth Error [m]", series="src_z", iteration=trial_index, value=candidates["src_z"])
        # logger.report_scalar(title="Objective Function", series="bartlett", iteration=trial_index, value=raw_data["bartlett"][0])
        with mlflow.start_run(experiment_id=experiment_id, run_name=str(trial_index), nested=True) as run:
            run_id = run.info.run_id
            if i >= n_warmup:
                try:
                    client.log_metric(run_id, "covar_length_range", ax_client._generation_strategy._model.model._surrogate._model.covar_module.base_kernel.kernels._modules["0"].lengthscale, step=trial_index)
                    client.log_metric(run_id, "covar_length_depth", ax_client._generation_strategy._model.model._surrogate._model.covar_module.base_kernel.kernels._modules["1"].lengthscale, step=trial_index)
                except:
                    client.log_metric(run_id, "covar_length_range", ax_client._generation_strategy._model.model.surrogate._model.covar_module.base_kernel.kernels._modules["0"].lengthscale, step=trial_index)
                    client.log_metric(run_id, "covar_length_depth", ax_client._generation_strategy._model.model.surrogate._model.covar_module.base_kernel.kernels._modules["1"].lengthscale, step=trial_index)
            
            ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)

            best_parameters, values = ax_client.get_best_parameters(use_model_predictions=False)
            means, covariances = values
            client.log_metric(run_id, "best_range", best_parameters["rec_r"], step=trial_index)
            client.log_metric(run_id, "best_depth", best_parameters["src_z"], step=trial_index)
            client.log_metric(run_id, "best_obj", means["bartlett"], step=trial_index)

                    
    data, layout = ax_client.get_contour_plot()
    render(ax_client.get_contour_plot())
    render(ax_client.get_optimization_trace())
    
        
    
    # Log current trial parameters & metric
    
    # logger.report_scalar(title="BOGP", series="")
    
    # [client.log_metric(run_id, k, v, step=trial_index) for k, v in candidates.items()]
    # client.log_metric(run_id, "bartlett", raw_data["bartlett"][0], step=trial_index)
    # # Log best trial parameters & metric
    # [client.log_metric(run_id, "best " + k, v, step=trial_index) for k, v in best_parameters.items()]
    # client.log_metric(run_id, "best bartlett", means["bartlett"], step=trial_index)

    # # Log hyperparameters
    
    #     try:
    #         client.log_metric(run_id, "lengthscale_rec_r", ax_client._generation_strategy._model.model._surrogate._model.covar_module.base_kernel.kernels._modules["0"].lengthscale, step=trial_index)
    #         client.log_metric(run_id, "lengthscale_src_z", ax_client._generation_strategy._model.model._surrogate._model.covar_module.base_kernel.kernels._modules["1"].lengthscale, step=trial_index)
    #     except:
    #         client.log_metric(run_id, "lengthscale_rec_r", ax_client._generation_strategy._model.model.surrogate._model.covar_module.base_kernel.kernels._modules['0'].lengthscale, step=trial_index)
    #         client.log_metric(run_id, "lengthscale_src_z", ax_client._generation_strategy._model.model.surrogate._model.covar_module.base_kernel.kernels._modules['1'].lengthscale, step=trial_index)
                
        



# with mlflow.start_run(experiment_id=experiment_id, run_name=datetime.now().strftime("%Y-%m-%d %H:%M:%S")) as main_run:
    
#     for i in range(n_total):

#         candidates, trial_index = ax_client.get_next_trial()
#         parameters = {"env_parameters": env_parameters, "K": K, "frequencies": frequencies} | candidates

#         raw_data = objective_function(parameters)
#         ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)
        
#         best_parameters, values = ax_client.get_best_parameters()
#         means, covariances = values

                
#         with mlflow.start_run(experiment_id=experiment_id, run_name=str(trial_index), nested=True) as run:
#             run_id = run.info.run_id
#             # Log current trial parameters & metric
#             [client.log_metric(run_id, k, v, step=trial_index) for k, v in candidates.items()]
#             client.log_metric(run_id, "bartlett", raw_data["bartlett"][0], step=trial_index)
#             # Log best trial parameters & metric
#             [client.log_metric(run_id, "best " + k, v, step=trial_index) for k, v in best_parameters.items()]
#             client.log_metric(run_id, "best bartlett", means["bartlett"], step=trial_index)

#             # Log hyperparameters
#             if i >= n_warmup:
#                 try:
#                     client.log_metric(run_id, "lengthscale_rec_r", ax_client._generation_strategy._model.model._surrogate._model.covar_module.base_kernel.kernels._modules["0"].lengthscale, step=trial_index)
#                     client.log_metric(run_id, "lengthscale_src_z", ax_client._generation_strategy._model.model._surrogate._model.covar_module.base_kernel.kernels._modules["1"].lengthscale, step=trial_index)
#                 except:
#                     client.log_metric(run_id, "lengthscale_rec_r", ax_client._generation_strategy._model.model.surrogate._model.covar_module.base_kernel.kernels._modules['0'].lengthscale, step=trial_index)
#                     client.log_metric(run_id, "lengthscale_src_z", ax_client._generation_strategy._model.model.surrogate._model.covar_module.base_kernel.kernels._modules['1'].lengthscale, step=trial_index)
