#!/usr/bin/env python3

from functools import partial
from pathlib import Path
import sys

from botorch.acquisition.analytic import _log_ei_helper, _scaled_improvement
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
import torch
from tritonoa.at.models.kraken.runner import run_kraken
from tritonoa.sp.beamforming import beamformer
from tritonoa.sp.mfp import MatchedFieldProcessor
from tritonoa.sp.processing import simulate_covariance

sys.path.insert(0, str(Path(__file__).parents[1]))
from data.bo import common

sys.path.insert(0, str(Path(__file__).parents[3]))
from optimization import utils

plt.style.use(["science", "ieee", "std-colors"])

strategy_data = [
    {
        "label": "UCB",
        "subfig": "a",
        "seed": 2009,
        "bbox_to_anchor": (0.40, 0.71),
        "ylim": [0.4, 1.25],
        "acq_yscale": {"value": "linear"},
        "yticks": [0.5, 1.0],
        "kappa": 1.0,
    },
    {
        "label": "EI",
        "subfig": "b",
        "seed": 719,
        "bbox_to_anchor": (0.68, 0.71),
        "ylim": [-0.005, 0.08],
        "acq_yscale": {"value": "linear"},
        "yticks": [0.0, 0.05],
    },
    {
        "label": "LogEI",
        "subfig": "c",
        "seed": 719,
        "bbox_to_anchor": (0.68, 0.71),
        "ylim": [-1e6, -1],
        "acq_yscale": {"value": "symlog", "linthresh": 1e-3},
        "yticks": [-1e6, -1e4, -1e2],
    },
]


def expected_improvement(
    mu: np.ndarray, sigma: np.ndarray, best_f: float, xi: float = 0.0
) -> np.ndarray:
    """Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.

    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.

    Returns:
        Expected improvements at points X.
    """
    print("Computing EI")
    with np.errstate(divide="warn"):
        imp = mu - best_f - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


def log_expected_improvement(
    mu: np.ndarray, sigma: np.ndarray, best_f: float, xi: float = 0.0
) -> np.ndarray:
    """Computes the log EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.

    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.

    Returns:
        Log expected improvements at points X.
    """
    print("Computing LogEI")
    u = _scaled_improvement(mu, sigma, best_f, False)
    log_ei = _log_ei_helper(torch.tensor(u)) + np.log(torch.tensor(sigma))
    return log_ei.detach().numpy()


def upper_confidence_bound(
    mu: np.ndarray, sigma: np.ndarray, kappa: float = 1.0
) -> np.ndarray:
    """Computes the UCB at points based on mean and covariance functions
    from GP.

    Args:
        mu: Mean of the GP at points X.
        sigma: Standard deviation of the GP at points X.
        kappa: Exploitation-exploration trade-off parameter.

    Returns:
        UCB at points X.
    """
    print("Computing UCB")
    return mu + kappa * sigma


def get_gp_data(seed: int) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    x, y, x_t, f = initialize_data(seed=seed)
    kernel = Matern(length_scale_bounds=(1e-1, 1.0), nu=2.5)
    gpr = GaussianProcessRegressor(kernel=kernel).fit(
        x.reshape(-1, 1), y.reshape(-1, 1)
    )
    mu, sigma = gpr.predict(x_t.reshape(-1, 1), return_std=True)
    return x, y, x_t, f, mu, sigma


def initialize_data(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    environment = utils.load_env_from_json(
        common.SWELLEX96Paths.main_environment_data_sim
    )
    freq = common.FREQ
    rec_r_true = 1.0
    src_z_true = 60.0
    true_parameters = {
        "rec_r": rec_r_true,
        "src_z": src_z_true,
    }

    K = simulate_covariance(
        runner=run_kraken,
        parameters=environment | true_parameters,
        freq=freq,
    )

    MFP = MatchedFieldProcessor(
        runner=run_kraken,
        covariance_matrix=K,
        freq=freq,
        parameters=environment | {"src_z": src_z_true},
        beamformer=partial(beamformer, atype="cbf_ml"),
    )
    num_rvec = 200
    dr = 0.25
    rec_r_lim = (rec_r_true - dr, rec_r_true + dr)

    x_test = np.linspace(rec_r_lim[0], rec_r_lim[1], num_rvec)
    y_true = MFP({"rec_r": x_test})

    n_samples = 10
    random = np.random.default_rng(seed)
    x = random.uniform(rec_r_lim[0], rec_r_lim[1], size=n_samples)
    y = MFP({"rec_r": x})

    return x, y, x_test, y_true


def plot_bo_example() -> plt.Figure:
    fig = plt.figure(figsize=(3.5, 7.0))

    outer_gs = gridspec.GridSpec(3, 1, figure=fig, height_ratios=[1, 1, 1])

    for i, strategy in enumerate(strategy_data):
        print(strategy["label"])

        x, y, x_t, f, mu, sigma = get_gp_data(seed=strategy["seed"])
        lcb = mu - 2 * sigma
        ucb = mu + 2 * sigma
        if strategy["label"] == "UCB":
            alpha = upper_confidence_bound(mu, sigma, strategy["kappa"])
        if strategy["label"] == "EI":
            alpha = expected_improvement(-mu, sigma, np.max(-y))
        if strategy["label"] == "LogEI":
            alpha = log_expected_improvement(mu, sigma, np.min(y))
        x_next = x_t[np.argmax(alpha)]
        y_next = f[np.argmax(alpha)]

        xlim = (min(x_t), max(x_t))

        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer_gs[i], height_ratios=[3, 1], hspace=0.0
        )
        ax0 = fig.add_subplot(inner_gs[0])
        # True function
        ax0.plot(x_t, f, color="k", label="$\phi(\mathbf{m})$")
        # Observed data
        ax0.scatter(
            x, y, color="k", marker="o", facecolor="none", label="$\\boldsymbol{\phi}$"
        )
        # Posterior mean
        ax0.plot(x_t, mu, color="k", linestyle="--", label="$\mu(\mathbf{m})$")
        # Posterior uncertainty
        ax0.fill_between(
            x_t,
            lcb,
            ucb,
            color="k",
            alpha=0.2,
            label="$\pm 2\sigma(\mathbf{m})$",
        )
        # Next sample
        markerline, _, _ = ax0.stem(
            x_next,
            y_next,
            markerfmt="rD",
            linefmt=":",
            basefmt=" ",
            bottom=-0.2,
            label="$\phi(\mathbf{m}_{t+1})$",
        )
        markerline.set_markerfacecolor("none")
        ax0.set_xlim(xlim)
        ax0.set_ylim(-0.1, 1.1)
        ax0.set_xticklabels([])
        ax0.set_ylabel("$\phi(\mathbf{m})$")
        ax0.legend(bbox_to_anchor=strategy["bbox_to_anchor"], prop={"size": 7})
        ax0.text(
            -0.14,
            1.0,
            f"({strategy['subfig']})",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax0.transAxes,
            size=10,
        )

        # Acquisition function
        ax1 = fig.add_subplot(inner_gs[1])
        ax1.plot(x_t, alpha, color="k", label="Acquisition function")
        ax1.axvline(x_next, color="k", linestyle=":", label="Next sample")
        ax1.set_xlim(xlim)
        ax1.set_ylim(strategy["ylim"])
        ax1.set_yscale(**strategy["acq_yscale"])
        ax1.set_yticks(strategy["yticks"])
        ax1.set_xlabel("$\mathbf{m} = r_\mathrm{src}$ [km]", labelpad=0)
        ax1.set_ylabel(f"{strategy['label']}\n$\\alpha(\phi(\mathbf{{m}}))$")

    return fig


def main() -> plt.Figure:
    return plot_bo_example()


if __name__ == "__main__":
    fig = main()
    fig.savefig("bo_ucb_example.png", bbox_inches="tight")
