#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from string import ascii_lowercase
import sys
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scienceplots

sys.path.insert(0, str(Path(__file__).parents[1]))
from data.bo import common, helpers

plt.style.use(["science", "ieee", "std-colors"])


MODES = "Simulated"
# MODES = "Experimental"
N_INIT = 200
XLIM = (0, 3000)
if MODES == "Simulated":
    YLIM = (1e-8, 1.0)
    YTICKS = [1e-8, 1e-6, 1e-4, 1e-2, 1.0]
if MODES == "Experimental":
    YLIM = (0.05, 1.0)
    YTICKS = [0.05, 0.1, 0.5, 1.0]



def plot_wall_time(df: pd.DataFrame, ax: Optional[plt.Axes] = None):
    if ax is None:
        ax = plt.gca()

    for strategy in df["Strategy"].unique():
        for seed in df["seed"].unique():
            if strategy not in ["Sobol", "Random"]:
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
                alpha=0.25,
                linewidth=0.35,
            )

    return ax


def plot_est_dist(df: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        ax = plt.gca()

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


def plot_equiv_time(data: list[pd.DataFrame]) -> plt.Figure:
    fig, axs = plt.subplots(
        ncols=2,
        figsize=(3, 1.5),
        gridspec_kw={"wspace": 0.1, "width_ratios": [0.7, 0.3]},
    )

    ax = axs[0]
    for df in data:
        ax = plot_wall_time(df, ax)
    ax.set_xlim(XLIM)
    ax.set_xlabel("Wall time [s]")
    ax.set_ylabel("$\hat{\phi}(\mathbf{m})$")

    ax = axs[1]
    for df in data:
        ax = plot_est_dist(df, ax=ax)

    for i, ax in enumerate(axs):
        ax.text(0, 1.05, f"({ascii_lowercase[i]})", transform=ax.transAxes)
        ax.grid(True, linestyle="dotted")
        ax.set_yscale("log")
        ax.set_ylim(YLIM)
        ax.set_yticks(YTICKS)
        ax.set_yticklabels(YTICKS) if i == 0 else ax.set_yticklabels([])

    return fig


def main():
    path = common.SWELLEX96Paths.outputs / "runs"
    data_sobol = helpers.load_data(
        path, "sim_sobol_50k/*.npz", common.SEARCH_SPACE, common.TRUE_VALUES
    )
    data_ei = helpers.load_data(
        path, "sim_ei/*.npz", common.SEARCH_SPACE, common.TRUE_VALUES
    )
    data = [
        data_ei,
        data_sobol,
        # data_rand,
    ]
    fig = plot_equiv_time(data)
    return fig


if __name__ == "__main__":
    main()
