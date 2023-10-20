#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from dataclasses import dataclass
import logging
import time

import botorch
from botorch.acquisition.analytic import ProbabilityOfImprovement
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch

import common, helpers


@dataclass
class PILoopArgs:
    dim: int
    budget: int = 500
    n_init: int = 100
    dtype: torch.dtype = torch.double
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_restarts: int = 40
    raw_samples: int = 1024
    seed: int = 0


def loop(
    objective: callable,
    budget,
    dim,
    n_init,
    dtype,
    device,
    num_restarts,
    raw_samples,
    seed: int = 0,
    *args,
    **kwargs,
) -> tuple[torch.tensor, torch.tensor]:
    logging.info(f"Running GP/PI on {device.type.upper()}.")

    start = time.time()

    X = helpers.get_initial_points(dim, n_init, dtype, device, seed)
    Y = -torch.tensor(objective(X.detach().cpu().numpy()), dtype=dtype, device=device)

    stop = time.time() - start
    times = [stop / n_init for _ in range(n_init)]

    logging.info(f"{n_init} warmup trials complete.")
    helpers.log_best_value_and_parameters(
        X.detach().cpu().numpy(), -Y.detach().cpu().numpy(), common.SEARCH_SPACE
    )
    logging.info("Commencing Bayesian optimization.")

    # Disable input scaling checks as we normalize to [-1, 1]
    with botorch.settings.validate_input_scaling(False):
        while len(Y) < budget:
            start = time.time()

            train_Y = (Y - Y.mean()) / Y.std()
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            model = SingleTaskGP(X, train_Y, likelihood=likelihood)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            optimizer = torch.optim.AdamW([{"params": model.parameters()}], lr=0.1)
            model.train()
            model.likelihood.train()
            for _ in range(50):
                optimizer.zero_grad()
                output = model(X)
                loss = -mll(output, train_Y.squeeze())
                loss.backward()
                optimizer.step()

            # Create a batch
            pi = ProbabilityOfImprovement(model, train_Y.max())
            candidate, acq_value = optimize_acqf(
                pi,
                bounds=torch.stack(
                    [
                        -torch.ones(dim, dtype=dtype, device=device),
                        torch.ones(dim, dtype=dtype, device=device),
                    ]
                ),
                q=1,
                num_restarts=num_restarts,
                raw_samples=raw_samples,
            )
            Y_next = -torch.tensor(
                objective(candidate.detach().cpu().numpy()),
                dtype=dtype,
                device=device,
            )

            # Append data
            X = torch.cat((X, candidate), axis=0)
            Y = torch.cat((Y, Y_next), axis=0)

            times.append(time.time() - start)

            # Print current status
            print("-" * 100)
            logging.info(f"GP/EI | Trial {len(X)}")
            helpers.log_current_value_and_parameters(
                X.detach().cpu().numpy(), -Y.detach().cpu().numpy(), common.SEARCH_SPACE
            )
            helpers.log_best_value_and_parameters(
                X.detach().cpu().numpy(), -Y.detach().cpu().numpy(), common.SEARCH_SPACE
            )

    return X, -Y, times
