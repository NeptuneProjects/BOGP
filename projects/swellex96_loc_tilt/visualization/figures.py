#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import seaborn as sns

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
    fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0})

    # Test points
    x = np.linspace(0, 1, 100)
    # True function
    f = x**2
    # Observed data
    x_t = np.random.rand(5)
    y = x_t**2
    # Posterior mean
    mu = x
    # Posterior uncertainty
    lcb = mu - 0.1
    ucb = mu + 0.1
    # Acquisition function
    alpha = 0.5 * np.sin(x * 2 * np.pi) + 0.5
    # Next sample
    x_next = x[np.argmax(alpha)]
    y_next = x_next**2

    xlim = (min(x), max(x))

    ax = axs[0]
    # True function
    ax.plot(x, f, color="k", label="$\phi(\mathbf{m})$")
    # Observed data
    ax.scatter(x_t, y, color="k", label="$\mathbf{y}$")
    # Posterior mean
    ax.plot(x, mu, color="k", linestyle="--", label="$\mu(\mathbf{m})$")
    # Posterior uncertainty
    ax.fill_between(
        x,
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
    ax.legend(loc="upper left")

    ax = axs[1]
    ax.plot(x, alpha, color="k", label="Acquisition function")
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
        "s": 5,
        "linewidth": 0.5,
        "alpha": 0.75,
    }
    error_plot_kwargs = {
        "y": 0.0,
        "color": "lightgray",
        "linewidth": 0.5,
        # "alpha": 0.25,
        "zorder": 10,
    }
    true_plot_kwargs = {
        "color": "k",
        "linewidth": 0.5,
        "zorder": 10,
    }
    data_plot_kwargs = {
        "gpei": common_data_plot_kwargs | {"color": "black", "marker": "o", "zorder": 40},
        "grid": common_data_plot_kwargs | {"color": "red", "marker": "D", "zorder": 30},
        "sobol": common_data_plot_kwargs | {"color": "blue", "marker": "s", "zorder": 20},
    }

    fig, axs = plt.subplots(3, 2, gridspec_kw={"hspace": 0.0, "wspace": 0.0})

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
    ax.set_ylim(0, 6)
    ax.set_xticklabels([])
    ax.set_ylabel("Range [km]")

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
    ax.set_ylim(100, 20)
    ax.set_xticklabels([])
    ax.set_ylabel("Depth [m]")

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
    ax.set_ylim(-4, 4)
    ax.set_ylabel("Tilt [$^\circ$]")
    ax.set_xlabel("Time Step")

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
    ax.set_xticklabels([])
    ax.yaxis.tick_right()

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
    ax.set_ylim(-40, 40)
    ax.set_xticklabels([])
    ax.yaxis.tick_right()

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
    ax.set_ylim(-4, 4)
    ax.yaxis.tick_right()
    ax.set_xlabel("Time Step")

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
        "gpei": "Sobol+GP/EI",
        "grid": "Grid",
        "sobol": "Sobol",
    }

    df_gpei = pd.read_csv(loadpath / "aggregated_results.csv")
    df_gpei = df_gpei[df_gpei["strategy"] == "gpei"]
    df_gpei = df_gpei[~df_gpei["param_time_step"].isin(skip_steps)]

    GRID_SERIAL = "serial_010"
    df_grid = pd.read_csv(loadpath.parent.parent / GRID_SERIAL / "results" / "aggregated_results.csv")
    df_grid = df_grid[df_grid["strategy"] == "grid"]
    df_grid = df_grid[~df_grid["param_time_step"].isin(skip_steps)]

    SOBOL_SERIAL = "serial_011"
    df_sobol = pd.read_csv(loadpath.parent.parent / SOBOL_SERIAL / "results" / "aggregated_results.csv")
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
        "--figure", "-f", default="2a", type=str, help="Figure number to create."
    )
    args = parser.parse_args()
    main(args)
