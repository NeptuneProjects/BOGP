#!/usr/bin/env python3

from functools import partial
from pathlib import Path
import sys

from botorch.acquisition.analytic import _log_ei_helper, _scaled_improvement
import matplotlib.pyplot as plt
import numpy as np
import scienceplots
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
    u = _scaled_improvement(mu, sigma, best_f, False)
    log_ei = _log_ei_helper(torch.tensor(u)) + np.log(torch.tensor(sigma))
    return log_ei.detach().numpy()


def get_gp_data() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    x, y, x_t, f = initialize_data()
    kernel = Matern(length_scale_bounds=(1e-1, 1.0), nu=2.5)
    gpr = GaussianProcessRegressor(kernel=kernel).fit(
        x.reshape(-1, 1), y.reshape(-1, 1)
    )
    mu, sigma = gpr.predict(x_t.reshape(-1, 1), return_std=True)
    alpha = log_expected_improvement(mu, sigma, np.min(y))
    return x, y, x_t, f, mu, sigma, alpha


def initialize_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    num_rvec = 400
    dr = 0.25
    rec_r_lim = (rec_r_true - dr, rec_r_true + dr)

    x_test = np.linspace(rec_r_lim[0], rec_r_lim[1], num_rvec)
    y_true = MFP({"rec_r": x_test})

    n_samples = 10
    random = np.random.default_rng(719)
    x = random.uniform(rec_r_lim[0], rec_r_lim[1], size=n_samples)
    y = MFP({"rec_r": x})

    return x, y, x_test, y_true


def plot_bo_example() -> plt.Figure:
    fig, axs = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0}, figsize=(3.5, 2.0)
    )
    x, y, x_t, f, mu, sigma, alpha = get_gp_data()
    lcb = mu - 2 * sigma
    ucb = mu + 2 * sigma
    # alpha /= alpha.max()
    x_next = x_t[np.argmax(alpha)]
    y_next = f[np.argmax(alpha)]

    xlim = (min(x_t), max(x_t))

    ax = axs[0]
    # True function
    ax.plot(x_t, f, color="k", label="$\phi(\mathbf{m})$")
    # Observed data
    ax.scatter(
        x, y, color="k", marker="o", facecolor="none", label="$\\boldsymbol{\phi}$"
    )
    # Posterior mean
    ax.plot(x_t, mu, color="k", linestyle="--", label="$\mu(\mathbf{m})$")
    # Posterior uncertainty
    ax.fill_between(
        x_t,
        lcb,
        ucb,
        color="k",
        alpha=0.2,
        label="$\pm 2\sigma(\mathbf{m})$",
    )
    # Next sample
    markerline, _, _ = ax.stem(
        x_next,
        y_next,
        markerfmt="rD",
        linefmt=":",
        basefmt=" ",
        bottom=-0.2,
        label="$\phi(\mathbf{m}_{t+1})$",
    )
    markerline.set_markerfacecolor("none")
    ax.set_xlim(xlim)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticklabels([])
    ax.set_ylabel("$\phi(\mathbf{m})$")
    ax.legend(bbox_to_anchor=(0.68, 0.71), prop={"size": 7})
    ax.text(
        -0.14,
        1.0,
        "(c)",
        horizontalalignment="center",
        verticalalignment="center",
        transform=ax.transAxes,
        size=10,
    )

    # Acquisition function
    ax = axs[1]
    ax.plot(x_t, alpha, color="k", label="Acquisition function")
    ax.axvline(x_next, color="k", linestyle=":", label="Next sample")
    ax.set_xlim(xlim)
    ax.set_xlabel("$\mathbf{m} = r_\mathrm{src}$ [km]")
    ax.set_ylim([-1e6, -1])
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_yticks([-1e6, -1e4, -1e2])
    ax.set_ylabel("LogEI\n$\\alpha(\phi(\mathbf{m}))$")

    return fig


def main() -> plt.Figure:
    return plot_bo_example()


if __name__ == "__main__":
    fig = main()
    fig.savefig("bo_example.pdf", bbox_inches="tight")
