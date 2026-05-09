# -*- coding:utf-8 -*-

import logging
import math
import time
from dataclasses import dataclass

import botorch
import common
import gpytorch.settings as gpt_settings
import helpers
import numpy as np
import torch
from botorch.acquisition import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

MAX_CHOLESKY_SIZE = float("inf")


@dataclass
class TurboLoopArgs:
    dim: int
    budget: int = 500
    n_init: int = 100
    batch_size: int = 1
    dtype: torch.dtype = torch.double
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_restarts: int = 40
    raw_samples: int = 1024
    seed: int = 0


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.8
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        """Post-initialize the state of the trust region."""
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def update_state(state: TurboState, Y_next: torch.Tensor) -> TurboState:
    """Update the state of the trust region based on the new function values."""
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def generate_batch(
    state: TurboState,
    model: SingleTaskGP,  # GP model
    X: torch.Tensor,  # Evaluated points on the domain [0, 1]^d
    Y: torch.Tensor,  # Function values
    batch_size: int,
    # n_candidates: int | None = None,  # Number of candidates for Thompson sampling
    num_restarts: int = 10,
    raw_samples: int = 512,
    acqf: str = "ei",  # "ei" or "ts"
    # dtype: torch.dtype = torch.double,
    # device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Generate a new batch of points."""
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0
    assert X.max() <= 1.0
    assert torch.all(torch.isfinite(Y))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    ei = qLogExpectedImprovement(model, Y.max())
    X_next, _ = optimize_acqf(
        ei,
        bounds=torch.stack([tr_lb, tr_ub]),
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    return X_next


def loop(
    objective: callable,
    dim,
    budget,
    n_init,
    batch_size,
    dtype,
    device,
    num_restarts,
    raw_samples,
    starting_points: torch.Tensor | None = None,
    seed: int = 0,
    *args,
    **kwargs,
) -> tuple[torch.tensor, torch.tensor, list[float]]:
    logging.info(f"Running TuRBO on {device.type.upper()}.")
    logging.info(f"Using batch size of {batch_size}.")

    start = time.time()

    if starting_points is not None:
        X = starting_points
    else:
        X = helpers.get_initial_points(dim, n_init, dtype, device, seed)
    Y = -torch.tensor(
        np.array(objective(X.detach().cpu().numpy())), dtype=dtype, device=device
    )

    stop = time.time() - start
    times = [stop / n_init for _ in range(n_init)]
    logging.info(f"{n_init} warmup trials complete.")
    helpers.log_best_value_and_parameters(
        X.detach().cpu().numpy(), -Y.detach().cpu().numpy(), common.SEARCH_SPACE
    )

    state = TurboState(dim=dim, batch_size=batch_size, best_value=max(Y).item())

    logging.info("Commencing Bayesian optimization.")
    trust_region_length = []

    # Disable input scaling checks as we normalize to [-1, 1]
    with botorch.settings.validate_input_scaling(False):
        while len(Y) < budget:
            start = time.time()

            train_Y = (Y - Y.mean()) / Y.std()
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=dim,
                    lengthscale_constraint=Interval(0.005, 4.0),
                )
            )
            model = SingleTaskGP(
                X, train_Y, likelihood=likelihood, covar_module=covar_module
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            with gpt_settings.max_cholesky_size(MAX_CHOLESKY_SIZE):
                fit_gpytorch_mll(mll)

                X_next = generate_batch(
                    state,
                    model,
                    X,
                    train_Y,
                    batch_size,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    acqf="ei",
                )

            Y_next = -torch.tensor(
                np.array(objective(X_next.detach().cpu().numpy())),
                dtype=dtype,
                device=device,
            )

            # Append data
            X = torch.cat((X, X_next), axis=0)
            Y = torch.cat((Y, Y_next), axis=0)

            state = update_state(state, Y_next)

            times.append(time.time() - start)

            # Print current status
            print("-" * 100)
            logging.info(f"TuRBO | Trial {len(X)}")
            helpers.log_current_value_and_parameters(
                X.detach().cpu().numpy(), -Y.detach().cpu().numpy(), common.SEARCH_SPACE
            )
            helpers.log_best_value_and_parameters(
                X.detach().cpu().numpy(), -Y.detach().cpu().numpy(), common.SEARCH_SPACE
            )
            logging.info(f"Trust region length: {state.length:.3f}")
            trust_region_length.append(state.length)

    return X, -Y, times, trust_region_length
