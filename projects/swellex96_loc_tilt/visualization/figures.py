#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from functools import partial
from pathlib import Path
import sys
from typing import Optional

from matplotlib.axes._axes import Axes
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
from scipy.stats import norm
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from tritonoa.at.models.kraken.runner import run_kraken
from tritonoa.sp.beamforming import beamformer
from tritonoa.sp.mfp import MatchedFieldProcessor
from tritonoa.sp.processing import simulate_covariance

sys.path.insert(0, str(Path(__file__).parents[1]))
from conf import common

sys.path.insert(0, str(Path(__file__).parents[3]))
from optimization import utils

plt.style.use(["science", "ieee"])

SERIAL = "serial_009"
SKIP_REGIONS = [
    list(range(37, 43)),
    list(range(87, 91)),
    list(range(95, 98)),
]
skip_steps = sum(SKIP_REGIONS, [])

loadpath = Path.cwd().parent / (
    f"data/swellex96_S5_VLA_loc_tilt/outputs/loc_tilt/experimental/"
    + SERIAL
    + "/results"
)
datapath = Path.cwd().parent / "data/swellex96_S5_VLA_loc_tilt/gps/source_tow.csv"
savepath = Path.cwd().parent / "reports/manuscripts/202309_ICASSP"


def figure_1() -> plt.Figure:
    fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0}, figsize=(3.5, 2.0))
    x, y, x_t, f, mu, sigma, alpha = get_gp_data()
    lcb = mu - 2 * sigma
    ucb = mu + 2 * sigma
    x_next = x_t[np.argmax(alpha)]
    y_next = f[np.argmax(alpha)]

    xlim = (min(x_t), max(x_t))

    ax = axs[0]
    # True function
    ax.plot(x_t, f, color="k", label="$\phi(\mathbf{m})$")
    # Observed data
    ax.scatter(x, y, color="k", marker="o", facecolor="none", label="$\mathbf{y}$")
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
        markerfmt="D",
        linefmt=":",
        basefmt=" ",
        bottom=-0.2,
        label="$y_{t+1}$",
    )
    markerline.set_markerfacecolor("none")
    ax.set_xlim(xlim)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticklabels([])
    ax.set_ylabel("$\phi(\mathbf{m})$")
    ax.legend(loc="lower right")

    ax = axs[1]
    ax.plot(x_t, alpha, color="k", label="Acquisition function")
    ax.axvline(x_next, color="k", linestyle=":", label="Next sample")
    ax.set_xlim(xlim)
    ax.set_xlabel("$\mathbf{m} = r_\mathrm{src}$ [km]")
    ax.set_ylabel("$\\alpha(\phi(\mathbf{m}))$")

    return fig


def figure_2() -> plt.Figure:
    df_tow, df_res = load_results()

    # Drop rows equal to SKIP_STEPS:
    df_tow_skip = df_tow[~df_tow.index.isin(skip_steps)]
    df_res = df_res[~df_res["Time Step"].isin(skip_steps)]
    NT = len(df_tow)
    xlim = (0, NT - 1)

    strategies = ["gpei", "grid", "sobol"]
    strategy_names = {
        "gpei": "Sobol+GP/EI",
        "grid": "Grid",
        "sobol": "Sobol",
    }
    common_data_plot_kwargs = {
        "facecolor": "none",
        "s": 10,
        "linewidth": 0.5,
        "alpha": 0.75,
    }
    error_plot_kwargs = {
        "y": 0.0,
        "color": "darkgray",
        "linewidth": 0.5,
        "zorder": 10,
    }
    true_plot_kwargs = {
        "color": "k",
        "linewidth": 0.5,
        "zorder": 10,
    }
    data_plot_kwargs = {
        "gpei": common_data_plot_kwargs
        | {"color": "black", "marker": "o", "zorder": 40},
        "grid": common_data_plot_kwargs | {"color": "red", "marker": "D", "zorder": 30},
        "sobol": common_data_plot_kwargs
        | {"color": "blue", "marker": "s", "zorder": 20},
    }
    line_kwargs = {
        "linestyle": "none",
        "markerfacecolor": "none",
        "markersize": 4,
        "markeredgewidth": 0.5,
    }
    ylabel_kwargs = {}
    xlabel_kwargs = {"ha": "right"}
    block_kwargs = {
        "color": "lightgray",
    }

    fig, axs = plt.subplots(
        3, 2, gridspec_kw={"hspace": 0.0, "wspace": 0.0}, figsize=(4.0, 2.29)
    )

    ax = axs[0, 0]
    ax.set_title("Estimate")
    ax.plot(
        range(NT),
        df_tow["Apparent Range [km]"],
        label="Range",
        **true_plot_kwargs,
    )
    for strategy in strategies:
        selection = df_res["strategy"] == strategy_names[strategy]
        ax.scatter(
            df_res.loc[selection, "Time Step"],
            df_res.loc[selection, "best_rec_r"],
            **data_plot_kwargs[strategy],
        )
    ax.set_xlim(xlim)
    ax.set_ylim(0.0, 6.0)
    [ax.fill_between(region, *ax.get_ylim(), **block_kwargs) for region in SKIP_REGIONS]
    adjust_subplotticklabels(ax, 0, -1)
    ax.set_xticklabels([])
    ax.set_ylabel("Range [km]", **ylabel_kwargs)

    ax = axs[1, 0]
    ax.plot(range(NT), df_tow["Apparent Depth [m]"], label="Depth", **true_plot_kwargs)
    for strategy in strategies:
        selection = df_res["strategy"] == strategy_names[strategy]
        ax.scatter(
            df_res.loc[selection, "Time Step"],
            df_res.loc[selection, "best_src_z"],
            **data_plot_kwargs[strategy],
        )
    ax.set_xlim(xlim)
    ax.set_ylim(110.0, 10.0)
    [ax.fill_between(region, *ax.get_ylim(), **block_kwargs) for region in SKIP_REGIONS]
    adjust_subplotticklabels(ax, 0, -1)
    ax.set_xticklabels([])
    ax.set_ylabel("Depth [m]", **ylabel_kwargs)

    ax = axs[2, 0]
    ax.plot(range(NT), df_tow["Apparent Tilt [deg]"], label="Tilt", **true_plot_kwargs)
    for strategy in strategies:
        selection = df_res["strategy"] == strategy_names[strategy]
        ax.scatter(
            df_res.loc[selection, "Time Step"],
            df_res.loc[selection, "best_tilt"],
            **data_plot_kwargs[strategy],
        )
    ax.set_xlim(xlim)
    ax.set_ylim(-5.0, 5.0)
    [ax.fill_between(region, *ax.get_ylim(), **block_kwargs) for region in SKIP_REGIONS]
    adjust_subplotticklabels(ax, 0, -1)
    ax.set_ylabel("Tilt [$^\circ$]", **ylabel_kwargs)
    ax.set_xlabel("Time Step", **xlabel_kwargs)

    ax = axs[0, 1]
    ax.set_title("Error")
    ax.axhline(**error_plot_kwargs)
    for strategy in strategies:
        selection = df_res["strategy"] == strategy_names[strategy]
        ax.scatter(
            df_res.loc[selection, "Time Step"],
            df_res.loc[selection, "best_rec_r"].values
            - df_tow_skip["Apparent Range [km]"].values,
            **data_plot_kwargs[strategy],
        )
    ax.set_xlim(xlim)
    ax.set_ylim(-0.5, 0.5)
    [ax.fill_between(region, *ax.get_ylim(), **block_kwargs) for region in SKIP_REGIONS]
    ax.set_xticklabels([])
    ax.yaxis.tick_right()
    adjust_subplotticklabels(ax, 0, -1)

    ax = axs[1, 1]
    ax.axhline(**error_plot_kwargs)
    for strategy in strategies:
        selection = df_res["strategy"] == strategy_names[strategy]
        ax.scatter(
            df_res.loc[selection, "Time Step"],
            df_res.loc[selection, "best_src_z"].values
            - df_tow_skip["Apparent Depth [m]"].values,
            **data_plot_kwargs[strategy],
        )
    ax.set_xlim(xlim)
    ax.set_ylim(-40.0, 40.0)
    [ax.fill_between(region, *ax.get_ylim(), **block_kwargs) for region in SKIP_REGIONS]
    ax.set_xticklabels([])
    ax.yaxis.tick_right()
    adjust_subplotticklabels(ax, 0, -1)

    ax = axs[2, 1]
    ax.axhline(**error_plot_kwargs)
    for strategy in strategies:
        selection = df_res["strategy"] == strategy_names[strategy]
        ax.scatter(
            df_res.loc[selection, "Time Step"],
            df_res.loc[selection, "best_tilt"].values
            - df_tow_skip["Apparent Tilt [deg]"].values,
            **data_plot_kwargs[strategy],
        )
    ax.set_xlim(xlim)
    ax.set_ylim(-4.0, 4.0)
    [ax.fill_between(region, *ax.get_ylim(), **block_kwargs) for region in SKIP_REGIONS]
    ax.set_xticklabels([])
    ax.yaxis.tick_right()
    adjust_subplotticklabels(ax, 0, -1)

    custom_lines = [
        Line2D([0], [0], color="k", linestyle="-", linewidth=0.5, label="True"),
        Line2D(
            [0],
            [0],
            color="k",
            marker="o",
            label="Sobol+GP/EI",
            **line_kwargs,
        ),
        Line2D(
            [0],
            [0],
            color="r",
            marker="D",
            label="Grid",
            **line_kwargs,
        ),
        Line2D(
            [0],
            [0],
            color="b",
            marker="s",
            label="Sobol",
            **line_kwargs,
        ),
    ]
    ax.legend(
        handles=custom_lines,
        labels=["True", "Sobol+GP/EI", "Grid", "Sobol"],
        loc="upper left",
        ncols=2,
        bbox_to_anchor=(-0.35, -0.2),
        frameon=True,
    )

    print(type(ax))

    return fig


def figure_3() -> plt.Figure:
    strategies = ["gpei", "grid", "sobol"]
    strategy_names = {
        "gpei": "Sobol+GP/EI",
        "grid": "Grid",
        "sobol": "Sobol",
    }
    block_kwargs = {
        "y1": -0.1,
        "y2": 1.1,
        "color": "lightgray",
    }
    data_kwargs = {
        "gpei": {"color": "black", "linestyle": "-", "zorder": 40},
        "grid": {"color": "red", "linestyle": "--", "zorder": 30},
        "sobol": {"color": "blue", "linestyle": ":", "zorder": 20},
    }

    _, df = load_results()
    df = df[~(df["strategy"] == "Sobol+GP/qEI")]
    df.loc[df["Time Step"].isin(skip_steps), "best_value"] = np.nan

    fig = plt.figure(figsize=(3.5, 1.0))
    ax = plt.gca()
    for strategy in strategies:
        view = df[df["strategy"] == strategy_names[strategy]]
        ax.plot(
            view["Time Step"],
            view["best_value"],
            label=strategy_names[strategy],
            **data_kwargs[strategy],
        )
    [ax.fill_between(region, **block_kwargs) for region in SKIP_REGIONS]
    ax.set_xlim(0, 125)
    ax.set_ylim(0.1, 0.9)
    ax.legend(frameon=True, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.35))
    ax.set_xlabel("Time Step")
    ax.set_ylabel("$\hat{\phi}(\mathbf{m})$")
    return fig


def figure_4() -> plt.Figure:
    strategies = ["gpei", "grid", "sobol"]
    colors = {
        "gpei": "black",
        "grid": "red",
        "sobol": "blue",
    }
    trace_kwargs = {
        "x": "run_time",
        "y": "best_value",
        "alpha": 0.1,
    }
    text_kwargs = {
        "x": 0.97,
        "y": 0.9,
        "ha": "right",
        "va": "top",
    }
    text_values = {
        "gpei": "Sobol+GP/EI: 64 trials",
        "grid": "Grid: 1008 trials",
        "sobol": "Sobol: 1024 trials",
    }

    df_gpei = pd.read_csv(loadpath / "aggregated_results.csv")
    df_gpei = df_gpei[df_gpei["strategy"] == "gpei"]
    df_gpei = df_gpei[~df_gpei["param_time_step"].isin(skip_steps)]

    GRID_SERIAL = "serial_010"
    df_grid = pd.read_csv(
        loadpath.parent.parent / GRID_SERIAL / "results" / "aggregated_results.csv"
    )
    df_grid = df_grid[df_grid["strategy"] == "grid"]
    df_grid = df_grid[~df_grid["param_time_step"].isin(skip_steps)]

    SOBOL_SERIAL = "serial_011"
    df_sobol = pd.read_csv(
        loadpath.parent.parent / SOBOL_SERIAL / "results" / "aggregated_results.csv"
    )
    df_sobol = df_sobol[df_sobol["strategy"] == "sobol"]
    df_sobol = df_sobol[~df_sobol["param_time_step"].isin(skip_steps)]

    df = pd.concat([df_gpei, df_grid, df_sobol], axis=0)

    fig, axs = plt.subplots(
        3, 1, gridspec_kw={"hspace": 0}, sharex=True, sharey=True, figsize=(3.5, 1.5)
    )
    for i, (strategy, ax) in enumerate(zip(strategies, axs)):
        selected_df = df[df["strategy"] == strategy]
        for step in selected_df["param_time_step"].unique():
            sns.lineplot(
                data=selected_df[selected_df["param_time_step"] == step],
                color=colors[strategy],
                ax=ax,
                **trace_kwargs,
            )
        ax.text(**text_kwargs, s=text_values[strategy], transform=ax.transAxes)
        if i < len(strategies) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.set_ylabel(None)

    ax.set_ylim(0, 1)
    ax.set_xlabel("Run time [s]")
    ax.set_ylabel("$\hat{\phi}(\mathbf{m})$")

    return fig


def adjust_subplotticklabels(
    ax: Axes, low: Optional[int] = None, high: Optional[int] = None
) -> None:
    ticklabels = ax.get_yticklabels()
    if low is not None:
        ticklabels[low].set_va("bottom")
    if high is not None:
        ticklabels[high].set_va("top")


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
    with np.errstate(divide="warn"):
        imp = mu - best_f - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0

    return ei


def get_gp_data() -> (
    tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]
):
    x, y, x_t, f = initialize_data()
    kernel = Matern(length_scale_bounds=(1e-1, 1.0), nu=2.5)
    gpr = GaussianProcessRegressor(kernel=kernel).fit(
        x.reshape(-1, 1), y.reshape(-1, 1)
    )
    mu, sigma = gpr.predict(x_t.reshape(-1, 1), return_std=True)
    alpha = expected_improvement(-mu, sigma, np.max(-y))
    return x, y, x_t, f, mu, sigma, alpha / alpha.max()


def initialize_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    environment = utils.load_env_from_json(common.SWELLEX96Paths.environment_data)
    freq = common.FREQ
    rec_r_true = 2.5
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
    dr = 1.0
    rec_r_lim = (rec_r_true - dr, rec_r_true + dr)

    x_test = np.linspace(rec_r_lim[0], rec_r_lim[1], num_rvec)
    y_true = MFP({"rec_r": x_test})

    n_samples = 10
    random = np.random.default_rng(719)
    x = random.uniform(rec_r_lim[0], rec_r_lim[1], size=n_samples)
    y = MFP({"rec_r": x})

    return x, y, x_test, y_true


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    df_tow = pd.read_csv(datapath)
    df_res = pd.read_csv(loadpath / "collated_results.csv")
    return df_tow, df_res


def main(args) -> None:
    fig = eval(f"figure_{args.figure}()")
    print(f"Size: {fig.get_size_inches()}")
    fig.savefig(savepath / f"figure_{args.figure}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create figures for the swellex96_loc_tilt project."
    )
    parser.add_argument(
        "--figure", "-f", default="1", type=str, help="Figure number to create."
    )
    args = parser.parse_args()
    main(args)
