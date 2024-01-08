#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from string import ascii_lowercase
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scienceplots

sys.path.insert(0, str(Path(__file__).parents[1]))
from data.bo import common, helpers

plt.style.use(["science", "ieee", "std-colors"])


MODES = ["Simulated", "Experimental"]
N_INIT = 200
YLIM = [(1e-8, 1.0), (0.07, 1.0)]
YTICKS = [[1e-8, 1e-6, 1e-4, 1e-2, 1.0], [0.07, 0.1, 0.5, 1.0]]
YTICKLABELS = [["$10^{-8}$", "$10^{-6}$", "$10^{-4}$", "$10^{-2}$", "$10^{0}$"], ["0.07", "0.1", "0.5", "1"]]


def plot_performance_history(
    df: pd.DataFrame, ax: Optional[plt.Axes] = None
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()

    for strategy in df["Strategy"].unique():
        if strategy != "Sobol":
            df_ = df[df["n_init"] == N_INIT]
        else:
            df_ = df
        dfp = df_.loc[df_["Strategy"] == strategy].pivot(
            index="Trial", columns="seed", values="best_obj"
        )

        mean = dfp.mean(axis=1)
        # std = dfp.std(axis=1)
        min_ = dfp.min(axis=1)
        max_ = dfp.max(axis=1)
        ax.plot(dfp.index, mean, color=common.STRATEGY_COLORS[strategy])
        ax.plot(
            dfp.index,
            min_,
            color=common.STRATEGY_COLORS[strategy],
            linestyle="-.",
            linewidth=0.5,
        )
        ax.plot(
            dfp.index,
            max_,
            color=common.STRATEGY_COLORS[strategy],
            linestyle="-.",
            linewidth=0.5,
        )

    return ax


def plot_wall_time(df: pd.DataFrame, ax: Optional[plt.Axes] = None):
    if ax is None:
        ax = plt.gca()

    for strategy in df["Strategy"].unique():
        for seed in df["seed"].unique():
            if strategy != "Sobol":
                dfp = df.loc[
                    (df["Strategy"] == strategy)
                    & (df["seed"] == seed)
                    & (df["n_init"] == N_INIT)
                ]
            else:
                dfp = df.loc[(df["Strategy"] == strategy) & (df["seed"] == seed)]
            ax.plot(
                dfp["wall_time"],
                dfp["best_obj"],
                color=common.STRATEGY_COLORS[strategy],
                alpha=0.5,
                linewidth=0.75,
            )

    return ax


def plot_est_dist(df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()

    # sel = (df["Trial"] == 500) & (
    #     (df["Strategy"] == "Sobol")
    #     | ((df["Strategy"] != "Sobol") & (df["n_init"] == N_INIT))
    # )
    sel = ((df["Strategy"] == "Sobol") & (df["Trial"] == 50000)) | (
        (df["Strategy"] != "Sobol") & (df["n_init"] == N_INIT) & (df["Trial"] == 500)
    )

    dfp = df.loc[sel].sort_values(by="Strategy")
    sns.boxplot(
        data=dfp,
        x="Strategy",
        y="best_obj",
        hue="Strategy",
        whis=[0, 100],
        width=0.6,
        palette=common.STRATEGY_COLORS,
        ax=ax,
    )
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    return ax


def performance_plot(data: list[pd.DataFrame]) -> plt.Figure:
    fig, axs = plt.subplots(
        nrows=2, ncols=3, figsize=(8, 3), gridspec_kw={"wspace": 0.05, "hspace": 0.2}
    )

    for i, df in enumerate(data):
        axrow = axs[i]
        ax = axrow[0]
        ax = plot_performance_history(df, ax=ax)
        ax.set_xlim(-5, 505)

        ax = axrow[1]
        ax = plot_wall_time(df, ax=ax)
        # ax.set_xscale("log")
        ax.set_xlim(-10, 3000)

        ax = axrow[2]
        ax = plot_est_dist(df, ax=ax)

    for i, axrow in enumerate(axs):
        for j, ax in enumerate(axrow):
            ax.text(0, 1.05, f"({ascii_lowercase[3 * i + j]})", transform=ax.transAxes)
            if i == 1:
                if j == 0:
                    ax.set_xlabel("Trial")
                if j == 1:
                    ax.set_xlabel("Time [s]")
                if j == 2:
                    ax.set_xlabel("Strategy")

            ax.grid(True, linestyle="dotted")
            ax.set_yscale("log")
            ax.set_ylim(YLIM[i])
            ax.set_yticks(YTICKS[i])
            ax.set_xticklabels([]) if i != 1 else None
            ax.set_yticklabels([]) if j != 0 else ax.set_yticklabels(YTICKLABELS[i])
            ax.set_ylabel(
                "$\widehat{{\phi}}$", rotation=0
            ) if j == 0 else None
            ax.text(
                -0.28,
                0.5,
                f"{MODES[i]} data",
                transform=ax.transAxes,
                va="center",
                rotation=90,
            ) if j == 0 else None

    return fig


def main(prepend=""):
    path = common.SWELLEX96Paths.outputs / "runs"
    # data_sim = helpers.load_data(
    #     path, f"{prepend}sim_*/*.npz", common.SEARCH_SPACE, common.TRUE_VALUES
    # )
    data_exp = helpers.load_data(
        path, f"{prepend}exp_*/*.npz", common.SEARCH_SPACE, common.TRUE_VALUES
    )
    fig = performance_plot([data_exp, data_exp])
    # fig = performance_plot([data_exp, data_exp])
    return fig


if __name__ == "__main__":
    main()
