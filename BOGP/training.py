#!/usr/bin/env python3

from pathlib import Path

from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.test_functions import Griewank
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
import matplotlib.pyplot as plt
import numpy as np
import torch

import swellex


def get_bounds(search_parameters):
    bounds = torch.zeros(2, len(search_parameters))
    for i, parameter in enumerate(search_parameters):
        bounds[:, i] = torch.tensor(parameter["bounds"])
    return bounds


def get_test_points(bounds, num_samples, dtype=torch.double):
    d = bounds.size(1)
    if d == 1:
        bounds = bounds.squeeze()
        X = torch.linspace(bounds[0], bounds[1], num_samples).unsqueeze(-1).to(dtype)
    else:
        if isinstance(num_samples, int):
            num_samples = [num_samples for _ in range(d)]
        x = []
        for i in range(d):
            x.append(
                torch.linspace(bounds[0, i], bounds[1, i], num_samples[i]).to(dtype)
            )

        Xm = torch.meshgrid(*x, indexing="xy")
        X = torch.cat([x.flatten().unsqueeze(-1) for x in Xm], dim=1)
    return X


def generate_random_data(bounds, num_samples, dtype=torch.double):
    d = bounds.size(1)
    if d == 1:
        bounds = bounds.squeeze()
        X = (
            torch.distributions.uniform.Uniform(bounds[0], bounds[1])
            .sample([num_samples])
            .unsqueeze(-1)
            .to(dtype)
        )
    else:
        if isinstance(num_samples, int):
            num_samples = [num_samples for _ in range(d)]
        x = []
        for i in range(d):
            x.append(
                torch.distributions.uniform.Uniform(bounds[0, i], bounds[1, i])
                .sample([num_samples[i]])
                .to(dtype)
            )
        Xm = torch.meshgrid(*x, indexing="xy")
        X = torch.cat([x.flatten().unsqueeze(-1) for x in Xm], dim=1)
    return X


def generate_initial_data(bounds, n_train=10, n_test=None, dtype=torch.double):
    X_train = generate_random_data(bounds, n_train, dtype)
    y_train = branin(X_train).unsqueeze(-1)
    best_y = y_train.max()

    if n_test is not None:
        X_test = get_test_points(bounds, n_test)
        y_test = branin(X_test).unsqueeze(-1)
    else:
        X_test, y_test = None, None

    return X_train, y_train, best_y, X_test, y_test


def initialize_model(X_train, y_train, state_dict=None):
    model = SingleTaskGP(train_X=X_train, train_Y=y_train)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


def optimize_acqf_and_get_observation(X, acq_func):
    alpha = acq_func(X.unsqueeze(1))
    ind = torch.argmax(alpha)
    X_new = X[ind].detach().unsqueeze(-1).T
    y_new = branin(X_new).unsqueeze(-1)
    return X_new, y_new, alpha


def plot_training(
    X_test,
    y_actual,
    X_train,
    y_train,
    mean,
    lcb,
    ucb,
    alpha,
    alpha_prev=None,
    title=None,
):
    max_alpha = np.argmax(alpha)
    alpha /= alpha.max()
    if alpha_prev is not None:
        max_alpha_prev = np.argmax(alpha_prev)
        alpha_prev /= alpha_prev.max() 
    else:
        max_alpha_prev = None

    fig, axs = plt.subplots(
        nrows=2, facecolor="white", figsize=(8, 6), gridspec_kw={"hspace": 0}
    )

    ax = axs[0]
    ax.set_title(title)
    ax.plot(X_test, y_actual, color="tab:green", label="Actual f")
    ax.plot(X_test, mean, label="f(x)")
    ax.fill_between(X_test.squeeze(), lcb, ucb, alpha=0.25)
    if not max_alpha_prev:
        ax.scatter(X_train, y_train, c="k", marker="x", label="Samples", zorder=40)
    else:
        ax.scatter(
            X_train[:-1], y_train[:-1], c="k", marker="x", label="Samples", zorder=40
        )
        ax.scatter(
            X_train[-1], y_train[-1], c="r", marker="*", label="Samples", zorder=50
        )
    ax.axvline(X_test[max_alpha], color="k", linestyle="-")
    if max_alpha_prev:
        ax.axvline(X_test[max_alpha_prev], color="r", linestyle=":")

    ax.set_xticklabels([])
    ax.set_xlabel(None)
    ax.set_ylabel("$f(\mathbf{x})$", rotation=0, ha="right")

    ax = axs[1]
    ax.plot(X_test, alpha)
    ax.axvline(X_test[max_alpha], color="k", linestyle="-")
    if max_alpha_prev:
        ax.axvline(X_test[max_alpha_prev], color="r", linestyle=":")

    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("$\\alpha(\mathbf{x})$", rotation=0, ha="left")

    # plt.legend()
    plt.show()
    return fig


dtype = torch.double
torch.manual_seed(2009)

branin = Griewank(negate=True, dim=1)

search_space = [
    {"name": "x1", "bounds": [-10.0, 10.0]},
    # {
    #     "name": "x2",
    #     "bounds": [-5.0, 5.0]
    # },
    # {
    #     "name": "rec_r",
    #     "type": "range",
    #     "bounds": [0.01, 10.0],
    #     "value_type": "float",
    #     "log_scale": False,
    # },
    # {
    #     "name": "src_z",
    #     "type": "range",
    #     "bounds": [1.0, 200.0],
    #     "value_type": "float",
    #     "log_scale": False,
    # },
]
env_parameters = swellex.environment
bounds = get_bounds(search_space)

X_train, y_train, best_y, X_test, y_actual = generate_initial_data(
    bounds, n_train=5, n_test=1001, dtype=dtype
)
mll, model = initialize_model(X_train, y_train)


NUM_TRIALS = 3

mean = []
ucb = []
lcb = []
a = []

for trial in range(NUM_TRIALS):
    fit_gpytorch_model(mll)

    mll.eval()
    with torch.no_grad():
        posterior = mll.model(X_test)
        y_test = posterior.mean
        cov_test = posterior.confidence_region()

        mean.append(y_test.detach().cpu().numpy())
        lcb.append(cov_test[0].detach().cpu().numpy())
        ucb.append(cov_test[1].detach().cpu().numpy())

    EI = ExpectedImprovement(model=model, best_f=best_y)
    X_new, y_new, alpha = optimize_acqf_and_get_observation(X_test, EI)

    a.append(alpha.detach().cpu().numpy())

    if trial == 0:
        plot_training(
            X_test.detach().cpu().numpy(),
            y_actual.detach().cpu().numpy(),
            X_train.detach().cpu().numpy(),
            y_train.detach().cpu().numpy(),
            mean[-1],
            lcb[-1],
            ucb[-1],
            a[-1],
            title="Initialization",
        )
    else:
        plot_training(
            X_test.detach().cpu().numpy(),
            y_actual.detach().cpu().numpy(),
            X_train.detach().cpu().numpy(),
            y_train.detach().cpu().numpy(),
            mean[-1],
            lcb[-1],
            ucb[-1],
            a[-1],
            a[-2],
            title=f"Iteration {trial}",
        )

    X_train = torch.cat([X_train, X_new])
    y_train = torch.cat([y_train, y_new])
    best_y = y_train.max()

    mll, model = initialize_model(X_train, y_train, model.state_dict())


savepath = Path.cwd() / "scripts"
# np.save(savepath / "mean.npy", mean)
# np.save(savepath / "mean.npy", lcb)
# np.save(savepath / "mean.npy", ucb)
# np.save(savepath / "mean.npy", mean)
