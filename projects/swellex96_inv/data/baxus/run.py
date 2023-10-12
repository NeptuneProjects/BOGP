#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys
import warnings

import botorch
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.exceptions import ModelFittingError
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.test_functions import Branin
import gpytorch
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.quasirandom import SobolEngine

import baxus, gen, obj

sys.path.insert(0, str(Path(__file__).parents[2]))
from conf import common

warnings.filterwarnings(
    "ignore", message="A not p.d., added jitter of 1e-08 to the diagonal"
)

SMOKE_TEST = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.double
DUMMY_DIM = 50
TRUE_DIM = len(common.SEARCH_SPACE)
DIM = TRUE_DIM + DUMMY_DIM
BUDGET = 500
N_INIT = 100
NUM_RESTARTS = 40 if not SMOKE_TEST else 2
RAW_SAMPLES = 1024 if not SMOKE_TEST else 4
N_CANDIDATES = min(5000, max(2000, 200 * DIM)) if not SMOKE_TEST else 4
MAX_CHOLESKY_SIZE = float("inf")  # Always use Cholesky

branin = Branin(negate=True).to(device=DEVICE, dtype=DTYPE)


def branin_emb(x):
    """x is assumed to be in [-1, 1]^D"""
    lb, ub = branin.bounds
    return branin(lb + (ub - lb) * (x[..., :2] + 1) / 2)


def get_bounds_from_search_space(search_space: list) -> torch.tensor:
    return torch.tensor(
        [(d["bounds"][0], d["bounds"][1]) for d in search_space],
        dtype=DTYPE,
        device=DEVICE,
    )


def convert_tensor_to_parameters(x: torch.tensor) -> dict:
    return [
        {
            d["name"]: float(xs[i].detach().cpu().numpy())
            for i, d in enumerate(common.SEARCH_SPACE)
        }
        for xs in x
    ]


def evaluate_objective(objective: callable, parameters: list[dict]) -> list[float]:
    with ThreadPoolExecutor(max_workers=min(len(parameters), 64)) as executor:
        results = executor.map(objective, parameters)
    return [1 - result for result in results]


def objective(x: torch.tensor) -> list[float]:
    """x is assumed to be in [-1, 1]^D"""
    xs = x[:, :TRUE_DIM]
    objective = obj.get_objective()
    search_space = common.SEARCH_SPACE
    bounds = get_bounds_from_search_space(search_space)
    xs = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * (xs + 1) / 2
    parameters = convert_tensor_to_parameters(xs)
    return evaluate_objective(objective, parameters)


def main():
    print(f"Running on {DEVICE}")

    print("=" * 50)
    print("Running Baxus")
    state = baxus.BaxusState(dim=DIM, eval_budget=BUDGET - N_INIT)
    S = baxus.embedding_matrix(
        input_dim=state.dim, target_dim=state.d_init, device=DEVICE
    )

    X_baxus_target = gen.get_initial_points(state.d_init, N_INIT, DTYPE, DEVICE)
    X_baxus_input = X_baxus_target @ S
    Y_baxus = torch.tensor(objective(X_baxus_input), dtype=DTYPE, device=DEVICE)

    # print(y.shape, y)
    # Y_baxus = torch.tensor(
    #     [branin_emb(x) for x in X_baxus_input], dtype=DTYPE, device=DEVICE
    # ).unsqueeze(-1)
    # print(Y_baxus.shape, Y_baxus)

    # Disable input scaling checks as we normalize to [-1, 1]
    with botorch.settings.validate_input_scaling(False):
        for _ in range(BUDGET - N_INIT):  # Run until evaluation budget depleted
            # Fit a GP model
            train_Y = (Y_baxus - Y_baxus.mean()) / Y_baxus.std()
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
                X_baxus_target,
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
                    print("    Model-fitting error; fitting with Adam.")
                    optimizer = torch.optim.AdamW(
                        [{"params": model.parameters()}], lr=0.1
                    )

                    for _ in range(100):
                        optimizer.zero_grad()
                        output = model(X_baxus_target)
                        loss = -mll(output, train_Y.flatten())
                        loss.backward()
                        optimizer.step()

                # Create a batch
                X_next_target = gen.create_candidate(
                    state=state,
                    model=model,
                    X=X_baxus_target,
                    Y=train_Y,
                    dtype=DTYPE,
                    device=DEVICE,
                    n_candidates=N_CANDIDATES,
                    num_restarts=NUM_RESTARTS,
                    raw_samples=RAW_SAMPLES,
                    acqf="ts",
                )

            X_next_input = X_next_target @ S

            # Y_next = torch.tensor(
            #     [branin_emb(x) for x in X_next_input], dtype=DTYPE, device=DEVICE
            # ).unsqueeze(-1)
            Y_next = torch.tensor(objective(X_next_input), dtype=DTYPE, device=DEVICE)

            # Update state
            state = baxus.update_state(state=state, Y_next=Y_next)

            # Append data
            X_baxus_input = torch.cat((X_baxus_input, X_next_input), dim=0)
            X_baxus_target = torch.cat((X_baxus_target, X_next_target), dim=0)
            Y_baxus = torch.cat((Y_baxus, Y_next), dim=0)

            # Print current status
            print(
                f"Trial {len(X_baxus_input)} | d={len(X_baxus_target.T)} | Best value: {1 - state.best_value:.3} | TR length: {state.length:.3}"
            )

            if state.restart_triggered:
                state.restart_triggered = False
                print("Increasing target space dimensionality...")
                S, X_baxus_target = baxus.increase_embedding_and_observations(
                    S, X_baxus_target, state.new_bins_on_split, DEVICE
                )
                print(f"New dimensionality: {len(S)}")
                state.target_dim = len(S)
                state.length = state.length_init
                state.failure_counter = 0
                state.success_counter = 0

    print(f"Complete.")

    print("=" * 50)
    print("Running GP/EI")
    X_ei = gen.get_initial_points(TRUE_DIM, N_INIT, DTYPE, DEVICE)
    Y_ei = torch.tensor(objective(X_ei), dtype=DTYPE, device=DEVICE)
    # Y_ei = torch.tensor(
    #     [branin_emb(x) for x in X_ei], dtype=DTYPE, device=DEVICE
    # ).unsqueeze(-1)

    # Disable input scaling checks as we normalize to [-1, 1]
    with botorch.settings.validate_input_scaling(False):
        while len(Y_ei) < len(Y_baxus):
            train_Y = (Y_ei - Y_ei.mean()) / Y_ei.std()
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            model = SingleTaskGP(X_ei, train_Y, likelihood=likelihood)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            optimizer = torch.optim.AdamW([{"params": model.parameters()}], lr=0.1)
            model.train()
            model.likelihood.train()
            for _ in range(50):
                optimizer.zero_grad()
                output = model(X_ei)
                loss = -mll(output, train_Y.squeeze())
                loss.backward()
                optimizer.step()

            # Create a batch
            ei = ExpectedImprovement(model, train_Y.max())
            candidate, acq_value = optimize_acqf(
                ei,
                bounds=torch.stack(
                    [
                        -torch.ones(TRUE_DIM, dtype=DTYPE, device=DEVICE),
                        torch.ones(TRUE_DIM, dtype=DTYPE, device=DEVICE),
                    ]
                ),
                q=1,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
            )
            Y_next = torch.tensor(objective(candidate), dtype=DTYPE, device=DEVICE)
            # Y_next = torch.tensor(
            #     [branin_emb(x) for x in candidate], dtype=DTYPE, device=DEVICE
            # ).unsqueeze(-1)

            # Append data
            X_ei = torch.cat((X_ei, candidate), axis=0)
            Y_ei = torch.cat((Y_ei, Y_next), axis=0)

            # Print current status
            print(f"Trial {len(X_ei)} | Best value: {1 - Y_ei.max().item():.3}")

    print(f"Complete; best parameters: {X_ei[Y_ei.argmax()]}")

    print("=" * 50)
    print("Running Sobol")
    X_Sobol = (
        SobolEngine(TRUE_DIM, scramble=True, seed=0)
        .draw(len(X_baxus_input))
        .to(dtype=DTYPE, device=DEVICE)
        * 2
        - 1
    )
    Y_Sobol = torch.tensor(objective(X_Sobol), dtype=DTYPE, device=DEVICE)
    # Y_Sobol = torch.tensor(
    #     [branin_emb(x) for x in X_Sobol], dtype=DTYPE, device=DEVICE
    # ).unsqueeze(-1)

    print("Complete.")

    names = ["BAxUS", "EI", "Sobol"]
    runs = [Y_baxus, Y_ei, Y_Sobol]
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, run in zip(names, runs):
        fx = np.maximum.accumulate(run.cpu())
        # plt.plot(-fx + branin.optimal_value, marker="", lw=3)
        plt.plot(1 - fx, marker="", lw=3)

    plt.ylabel("Regret", fontsize=18)
    plt.xlabel("Number of evaluations", fontsize=18)
    plt.title(f"{DIM}D Bartlett Power", fontsize=24)
    plt.xlim([0, len(Y_baxus)])
    plt.yscale("log")

    plt.grid(True)
    plt.tight_layout()
    plt.legend(
        names + ["Global optimal value"],
        loc="lower center",
        bbox_to_anchor=(0, -0.08, 1, 1),
        bbox_transform=plt.gcf().transFigure,
        ncol=4,
        fontsize=16,
    )
    plt.show()


if __name__ == "__main__":
    main()
