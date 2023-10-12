# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
import logging
import math
from typing import Optional
import warnings

import botorch
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.exceptions import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
import torch
from torch.quasirandom import SobolEngine

import helpers

MAX_CHOLESKY_SIZE = float("inf")  # Always use Cholesky


@dataclass
class BAxUSLoopArgs:
    true_dim: int
    budget: int = 500
    n_init: int = 100
    dtype: torch.dtype = torch.double
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy_dim: Optional[int] = 100
    num_restarts: int = 40
    raw_samples: int = 1024
    dim: int = field(init=False)
    n_candidates: int = field(init=False)
    seed: int = 0

    def __post_init__(self):
        self.dim = self.true_dim + self.dummy_dim
        self.n_candidates = min(5000, max(2000, 200 * self.dim))


@dataclass
class BaxusState:
    dim: int
    eval_budget: int
    new_bins_on_split: int = 3
    d_init: int = float("nan")  # Note: post-initialized
    target_dim: int = float("nan")  # Note: post-initialized
    n_splits: int = float("nan")  # Note: post-initialized
    length: float = 0.8
    length_init: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    success_counter: int = 0
    success_tolerance: int = 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        n_splits = round(math.log(self.dim, self.new_bins_on_split + 1))
        self.d_init = 1 + np.argmin(
            np.abs(
                (1 + np.arange(self.new_bins_on_split))
                * (1 + self.new_bins_on_split) ** n_splits
                - self.dim
            )
        )
        self.target_dim = self.d_init
        self.n_splits = n_splits

    @property
    def split_budget(self) -> int:
        return round(
            -1
            * (self.new_bins_on_split * self.eval_budget * self.target_dim)
            / (self.d_init * (1 - (self.new_bins_on_split + 1) ** (self.n_splits + 1)))
        )

    @property
    def failure_tolerance(self) -> int:
        if self.target_dim == self.dim:
            return self.target_dim
        k = math.floor(math.log(self.length_min / self.length_init, 0.5))
        split_budget = self.split_budget
        return min(self.target_dim, max(1, math.floor(split_budget / k)))


def create_candidate(
    state: object,
    model: object,  # GP model
    X: torch.tensor,  # Evaluated points on the domain [-1, 1]^d
    Y: torch.tensor,  # Function values
    dtype: torch.dtype,
    device: torch.device,
    n_candidates: Optional[int] = None,  # Number of candidates for Thompson sampling
    num_restarts: int = 10,
    raw_samples: int = 512,
    acqf: str = "ts",  # "ei" or "ts"
):
    assert acqf in ("ts", "ei")
    assert X.min() >= -1.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.detach().view(-1)
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length, -1.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length, -1.0, 1.0)

    if acqf == "ts":
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = thompson_sampling(X_cand, num_samples=1)

    elif acqf == "ei":
        ei = ExpectedImprovement(model, Y.max())
        X_next, _ = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=1,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return X_next


def embedding_matrix(
    input_dim: int, target_dim: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    if (
        target_dim >= input_dim
    ):  # return identity matrix if target size greater than input size
        return torch.eye(input_dim, device=device, dtype=dtype)

    input_dims_perm = (
        torch.randperm(input_dim, device=device) + 1
    )  # add 1 to indices for padding column in matrix

    bins = torch.tensor_split(
        input_dims_perm, target_dim
    )  # split dims into almost equally-sized bins
    bins = torch.nn.utils.rnn.pad_sequence(
        bins, batch_first=True
    )  # zero pad bins, the index 0 will be cut off later

    mtrx = torch.zeros(
        (target_dim, input_dim + 1), dtype=dtype, device=device
    )  # add one extra column for padding
    mtrx = mtrx.scatter_(
        1,
        bins,
        2 * torch.randint(2, (target_dim, input_dim), dtype=dtype, device=device) - 1,
    )  # fill mask with random +/- 1 at indices

    return mtrx[:, 1:]  # cut off index zero as this corresponds to zero padding


def increase_embedding_and_observations(
    S: torch.Tensor,
    X: torch.Tensor,
    n_new_bins: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    assert X.size(1) == S.size(0), "Observations don't lie in row space of S"

    S_update = S.clone()
    X_update = X.clone()

    for row_idx in range(len(S)):
        row = S[row_idx]
        idxs_non_zero = torch.nonzero(row)
        idxs_non_zero = idxs_non_zero[torch.randperm(len(idxs_non_zero))].squeeze()

        non_zero_elements = row[idxs_non_zero].squeeze()

        # number of new bins is always less or equal than the contributing
        # input dims in the row minus one
        n_row_bins = min(n_new_bins, len(idxs_non_zero))

        # the dims in the first bin won't be moved
        new_bins = torch.tensor_split(idxs_non_zero, n_row_bins)[1:]
        elements_to_move = torch.tensor_split(non_zero_elements, n_row_bins)[1:]

        # pad the tuples of bins with zeros to apply _scatter
        new_bins_padded = torch.nn.utils.rnn.pad_sequence(new_bins, batch_first=True)
        els_to_move_padded = torch.nn.utils.rnn.pad_sequence(
            elements_to_move, batch_first=True
        )

        # submatrix to stack on S_update
        S_stack = torch.zeros(
            (n_row_bins - 1, len(row) + 1), device=device, dtype=dtype
        )

        # fill with old values (add 1 to indices for padding column)
        S_stack = S_stack.scatter_(1, new_bins_padded + 1, els_to_move_padded)

        # set values that were move to zero in current row
        S_update[row_idx, torch.hstack(new_bins)] = 0

        # set values that were move to zero in current row
        X_update = torch.hstack(
            (X_update, X[:, row_idx].reshape(-1, 1).repeat(1, len(new_bins)))
        )
        # stack onto S_update except for padding column
        S_update = torch.vstack((S_update, S_stack[:, 1:]))

    return S_update, X_update


def update_state(state, Y_next):
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


def loop(
    objective: callable,
    dim: int,
    budget: int,
    n_init: int,
    dtype: torch.dtype,
    device: torch.device,
    true_dim: int,
    num_restarts: int,
    raw_samples: int,
    n_candidates: int,
    seed: int = 0,
    *args,
    **kwargs,
)  -> tuple[torch.tensor, torch.tensor]:
    logging.info(f"Running BAxUS on {device}")
    state = BaxusState(dim=dim, eval_budget=budget - n_init)
    S = embedding_matrix(
        input_dim=state.dim, target_dim=state.d_init, dtype=dtype, device=device
    )

    X_target = helpers.get_initial_points(state.d_init, n_init, dtype, device, seed)
    X_input = X_target @ S
    Y = torch.tensor(
        objective(X_input.detach().cpu().numpy(), true_dim=true_dim),
        dtype=dtype,
        device=device,
    )

    logging.info(
        f"{n_init} warmup trials complete. | Best value: {1 - Y.max().item():.3}"
    )
    logging.info("Commencing Bayesian optimization.")

    # Disable input scaling checks as we normalize to [-1, 1]
    with botorch.settings.validate_input_scaling(False):
        for _ in range(budget - n_init):  # Run until evaluation budget depleted
            # Fit a GP model
            train_Y = (Y - Y.mean()) / Y.std()
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            covar_module = (
                ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
                    MaternKernel(
                        nu=2.5,
                        ard_num_dims=state.target_dim,
                        lengthscale_constraint=Interval(0.005, 10),
                    ),
                    outputscale_constraint=Interval(0.05, 10),
                )
            )
            model = SingleTaskGP(
                X_target,
                train_Y,
                covar_module=covar_module,
                likelihood=likelihood,
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)

            # Do the fitting and acquisition function optimization inside the Cholesky context
            with gpytorch.settings.max_cholesky_size(
                MAX_CHOLESKY_SIZE
            ), warnings.catch_warnings():
                # with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="A not p.d., added jitter of 1.0e-08 to the diagonal",
                )
                warnings.filterwarnings(
                    "ignore",
                    message="`scipy_minimize` terminated with status 3, displaying original message from `scipy.optimize.minimize`: ABNORMAL_TERMINATION_IN_LNSRCH",
                )
                # Fit the model
                try:
                    fit_gpytorch_mll(mll)
                except ModelFittingError:
                    # Right after increasing the target dimensionality, the covariance matrix becomes indefinite
                    # In this case, the Cholesky decomposition might fail due to numerical instabilities
                    # In this case, we revert to Adam-based optimization
                    logging.info("    Model-fitting error; fitting with Adam.")
                    optimizer = torch.optim.AdamW(
                        [{"params": model.parameters()}], lr=0.1
                    )

                    for _ in range(100):
                        optimizer.zero_grad()
                        output = model(X_target)
                        loss = -mll(output, train_Y.flatten())
                        loss.backward()
                        optimizer.step()

                # Create a batch
                X_next_target = create_candidate(
                    state=state,
                    model=model,
                    X=X_target,
                    Y=train_Y,
                    dtype=dtype,
                    device=device,
                    n_candidates=n_candidates,
                    num_restarts=num_restarts,
                    raw_samples=raw_samples,
                    acqf="ts",
                )

            X_next_input = X_next_target @ S

            # Y_next = torch.tensor(
            #     [branin_emb(x) for x in X_next_input], dtype=DTYPE, device=DEVICE
            # ).unsqueeze(-1)
            Y_next = torch.tensor(
                objective(X_next_input.detach().cpu().numpy(), true_dim=true_dim),
                dtype=dtype,
                device=device,
            )

            # Update state
            state = update_state(state=state, Y_next=Y_next)

            # Append data
            X_input = torch.cat((X_input, X_next_input), dim=0)
            X_target = torch.cat((X_target, X_next_target), dim=0)
            Y = torch.cat((Y, Y_next), dim=0)

            # Print current status
            logging.info(
                f"Trial {len(X_input)} | d={len(X_target.T)} | Best value: {1 - state.best_value:.3} | TR length: {state.length:.3}"
            )

            if state.restart_triggered:
                state.restart_triggered = False
                logging.info("Increasing target space dimensionality...")
                S, X_target = increase_embedding_and_observations(
                    S, X_target, state.new_bins_on_split, dtype, device
                )
                logging.info(f"New dimensionality: {len(S)}")
                state.target_dim = len(S)
                state.length = state.length_init
                state.failure_counter = 0
                state.success_counter = 0

    logging.info(f"Complete.")
    return X_input, Y
