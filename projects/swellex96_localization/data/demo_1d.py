from copy import deepcopy
from pathlib import Path
import sys
import warnings

from botorch import fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.exceptions import InputDataWarning
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc
import torch
from tritonoa.at.models.kraken.runner import run_kraken
from tritonoa.sp.beamforming import beamformer
from tritonoa.sp.mfp import MatchedFieldProcessor

sys.path.insert(0, str(Path(__file__).parents[2]))
from conf.swellex96.optimization.common import FREQ, SWELLEX96Paths
import optimization.utils as utils


def define_objective() -> callable:
    environment = utils.load_env_from_json(SWELLEX96Paths.environment_data) | {
        "tmpdir": "."
    }
    rec_r_true = 5.0
    src_z_true = 60.0
    true_parameters = {
        "rec_r": rec_r_true,
        "src_z": src_z_true,
    }

    K = utils.simulate_covariance(
        runner=run_kraken,
        parameters=environment | true_parameters,
        freq=FREQ,
    )

    return MatchedFieldProcessor(
        runner=run_kraken,
        covariance_matrix=K,
        freq=FREQ,
        parameters=environment | {"src_z": src_z_true},
        beamformer=beamformer,
    )


def define_search_space() -> list[dict]:
    return [
        {
            "name": "rec_r",
            "type": "range",
            "bounds": [4.0, 6.0],
            "value_type": "float",
            "log_scale": False,
        }
    ]


def get_bounds(search_parameters: list[dict]) -> torch.Tensor:
    bounds = torch.zeros(2, len(search_parameters))
    for i, parameter in enumerate(search_parameters):
        bounds[:, i] = torch.tensor(parameter["bounds"])
    return bounds


def generate_test_data(
    objective: callable, bounds: torch.Tensor, n_test: int
) -> tuple[torch.Tensor, torch.Tensor]:
    x_test = torch.linspace(
        bounds[0].item(), bounds[1].item(), n_test, dtype=torch.double
    ).reshape(-1, 1)
    y_actual = objective({"rec_r": x_test.detach().cpu().numpy().squeeze()})
    return x_test, torch.DoubleTensor(y_actual).unsqueeze(-1)


def generate_train_data(
    objective: callable, bounds: torch.Tensor, n_train: int
) -> tuple[torch.Tensor, torch.Tensor]:
    sampler = qmc.Sobol(d=1, scramble=True, seed=123)
    sample = sampler.random(n_train)
    x_train = torch.DoubleTensor(qmc.scale(sample, bounds[0].item(), bounds[1].item()))
    # x_train = torch.distributions.uniform.Uniform(*bounds).sample((n_train,)).double()
    y_train = objective({"rec_r": x_train.detach().cpu().numpy().squeeze()})
    return x_train, torch.DoubleTensor(y_train).unsqueeze(-1)


def initialize_model(
    x_train: torch.Tensor, y_train: torch.Tensor, state_dict=None
) -> tuple:
    model = SingleTaskGP(x_train, y_train)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


def optimize_acqf_and_get_observation(
    x: torch.Tensor, acq_func: callable, bounds: torch.Tensor, obj_func: callable
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    alpha = acq_func(x.unsqueeze(1))
    # candidates, _ = optimize_acqf(
    #     acq_function=acq_func, bounds=bounds, q=1, num_restarts=40, raw_samples=1024
    # )
    # x_new = candidates.detach()
    x_new = x[alpha.argmax()].detach().unsqueeze(-1)
    y_new = torch.DoubleTensor(obj_func({"rec_r": x_new})).unsqueeze(-1)
    return x_new, y_new, alpha


def main() -> None:
    torch.manual_seed(1)
    savepath = SWELLEX96Paths.outputs / "range_estimation" / "demo3"
    n_warmup = 8
    num_trials = 20
    n_test = 401

    objective = define_objective()
    bounds = get_bounds(define_search_space())

    x_test, y_actual = generate_test_data(objective, bounds, n_test)
    x_train, y_train = generate_train_data(objective, bounds, n_warmup)
    best_y = y_train.max()

    mll, model = initialize_model(x_train, y_train)

    mean = np.zeros((num_trials, n_test))
    ucb = np.zeros_like(mean)
    lcb = np.zeros_like(mean)
    alpha_array = np.zeros_like(mean)

    for trial in range(num_trials):
        fit_gpytorch_mll(mll)
        mll.eval()

        with torch.no_grad():
            posterior = mll.model(x_test)
            y_test = posterior.mean
            cov_test = posterior.confidence_region()

            mean[trial] = y_test.detach().cpu().numpy().squeeze()
            lcb[trial] = cov_test[0].detach().cpu().numpy().squeeze()
            ucb[trial] = cov_test[1].detach().cpu().numpy().squeeze()

        EI = ExpectedImprovement(model=model, best_f=best_y)
        x_new, y_new, alpha = optimize_acqf_and_get_observation(
            x_test, EI, bounds, objective
        )
        alpha_array[trial] = alpha.detach().cpu().numpy().squeeze()

        x_train = torch.cat([x_train, x_new])
        y_train = torch.cat([y_train, y_new])
        best_y = y_train.max()

        mll, model = initialize_model(x_train, y_train, model.state_dict())

    x_test_array = x_test.detach().cpu().numpy()
    y_actual_array = y_actual.detach().cpu().numpy()
    x_train_array = x_train.detach().cpu().numpy()
    y_train_array = y_train.detach().cpu().numpy()

    np.save(savepath / "X_test.npy", x_test_array)
    np.save(savepath / "y_actual.npy", y_actual_array)
    np.save(savepath / "X_train.npy", x_train_array)
    np.save(savepath / "y_train.npy", y_train_array)
    np.save(savepath / "mean.npy", mean)
    np.save(savepath / "lcb.npy", lcb)
    np.save(savepath / "ucb.npy", ucb)
    np.save(savepath / "alpha.npy", alpha_array)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=InputDataWarning)
        main()
