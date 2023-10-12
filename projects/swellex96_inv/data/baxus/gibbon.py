# -*- coding: utf-8 -*-

from dataclasses import dataclass
import logging

import botorch
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch

import helpers


@dataclass
class GIBBONLoopArgs:
    dim: int
    budget: int = 500
    n_init: int = 100
    dtype: torch.dtype = torch.double
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_restarts: int = 40
    raw_samples: int = 1024


def loop(
    objective: callable,
    budget,
    dim,
    n_init,
    dtype,
    device,
    num_restarts,
    raw_samples,
    *args,
    **kwargs,
) -> tuple[torch.tensor, torch.tensor]:
    logging.info(f"Running GP/qGIBBON on {device.type.upper()}.")
    X = helpers.get_initial_points(dim, n_init, dtype, device)
    Y = torch.tensor(objective(X.detach().cpu().numpy()), dtype=dtype, device=device)

    logging.info(
        f"{n_init} warmup trials complete. | Best value: {1 - Y.max().item():.3}"
    )
    logging.info("Commencing Bayesian optimization.")

    # Disable input scaling checks as we normalize to [-1, 1]
    with botorch.settings.validate_input_scaling(False):
        while len(Y) < budget:
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
            qGIBBON = qLowerBoundMaxValueEntropy(model, X)
            candidate, acq_value = optimize_acqf(
                qGIBBON,
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
            Y_next = torch.tensor(
                objective(candidate.detach().cpu().numpy()),
                dtype=dtype,
                device=device,
            )

            # Append data
            X = torch.cat((X, candidate), axis=0)
            Y = torch.cat((Y, Y_next), axis=0)

            # Print current status
            logging.info(f"Trial {len(X)} | Best value: {1 - Y.max().item():.3}")

    logging.info(f"Complete; best parameters: {X[Y.argmax()]}")
    return X, Y
