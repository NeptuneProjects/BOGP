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


MODES = ["Simulated", "Experimental"]
N_INIT = 64
N_RAND = 10000
YLIM = [(-0.01, 0.4), (0.0, 0.4)]
YTICKS = [[0.0, 0.2, 0.4], [0.0, 0.2, 0.4]]
YTICKLABELS = YTICKS
DIST_LABELS = [
    "Rand\n(100)",
    "Rand\n(10k)",
    "BO-\nUCB",
    "BO-\nEI",
    "BO-\nLogEI",
    "DE",
]


def performance_plot(data: list[pd.DataFrame], de_data: list[pd.DataFrame], n_init: int = 32) -> plt.Figure:
    fig, axs = plt.subplots(
        nrows=2, ncols=3, figsize=(8, 4), gridspec_kw={"wspace": 0.08, "hspace": 0.15}
    )
    for i, (df, de) in enumerate(zip(data, de_data)):
        df = helpers.split_random_results(df, 100)
        df = df.loc[df["Strategy"] != "Sobol"]

        axrow = axs[i]
        ax = axrow[0]
        ax = plot_performance_history(df, n_init, ax=ax)
        ax.set_xlim(1, 100)

        ax = axrow[1]
        ax = plot_wall_time(df, de, n_init, ax=ax)
        ax.set_xlim(-20, 1000)

        ax = axrow[2]
        ax = plot_est_dist(df, de, n_init, ax=ax)

    for i, axrow in enumerate(axs):
        for j, ax in enumerate(axrow):
            ax.text(0, 1.05, f"({ascii_lowercase[3 * i + j]})", transform=ax.transAxes)
            if i == 1:
                if j == 0:
                    ax.set_xlabel("Trial")
                if j == 1:
                    ax.set_xlabel("Time [s]")

            ax.grid(True, linestyle="dotted")
            ax.set_ylim(YLIM[i])
            ax.set_yticks(YTICKS[i])
            ax.set_yticklabels([]) if j != 0 else ax.set_yticklabels(YTICKLABELS[i])

            ax.set_xticklabels([]) if i != 1 else None
            ax.set_ylabel("$\widehat{{\phi}}$", rotation=0) if j == 0 else None
            (
                ax.text(
                    -0.28,
                    0.5,
                    f"{MODES[i]} data",
                    transform=ax.transAxes,
                    va="center",
                    rotation=90,
                )
                if j == 0
                else None
            )

            if i == 1 and j == 1:
                strategies = sorted(
                    list(df["Strategy"].unique()), key=common.SORTING_RULE.__getitem__
                ) + ["DE"]
                handles = [
                    plt.Line2D(
                        [0],
                        [0],
                        color=common.STRATEGY_COLORS[strategy],
                        label=strategy,
                        marker="o",
                        markersize=5,
                    )
                    for strategy in strategies
                ]

                ax.legend(
                    handles=handles,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.25),
                    ncol=6,
                    frameon=True,
                    fancybox=True,
                    borderaxespad=0.0,
                    title="Strategy",
                )

    return fig


def plot_performance_history(
    df: pd.DataFrame, n_init: int = 32, ax: Optional[plt.Axes] = None
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    strategies = sorted(
        list(df["Strategy"].unique()), key=common.SORTING_RULE.__getitem__
    )
    strategies.remove("Random (10k)")

    for strategy in strategies:
        if (strategy != "Random (100)") and (strategy != "Random (10k)"):
            df_ = df[df["n_init"] == n_init]
        else:
            df_ = df
        dfp = df_.loc[df_["Strategy"] == strategy].pivot(
            index="Trial", columns="seed", values="best_obj"
        )

        mean = dfp.mean(axis=1)
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
        ax.axvline(x=n_init, color="black", linestyle="--", linewidth=0.5)

    return ax


def plot_wall_time(df: pd.DataFrame, de: pd.DataFrame, n_init: int = 32, ax: Optional[plt.Axes] = None):

    SCATTER_KWARGS = {
        "s": 10,
        "marker": "o",
        "alpha": 0.25,
        "linewidths": 0.5,
        "zorder": 50,
    }

    if ax is None:
        ax = plt.gca()

    strategies = sorted(
        list(df["Strategy"].unique()), key=common.SORTING_RULE.__getitem__
    )

    sel = (
        ((df["Strategy"] == "Random (100)") & (df["Trial"] == 100))
        | ((df["Strategy"] == "Random (10k)") & (df["Trial"] == N_RAND))
        | (
            (df["Strategy"] != "Random")
            & (df["n_init"] == n_init)
            & (df["Trial"] == 100)
        )
    )
    df = df.loc[sel]

    for strategy in strategies:
        for seed in df["seed"].unique():
            if strategy == "Random (100)":
                dfp = df.loc[
                    (df["Strategy"] == strategy)
                    & (df["seed"] == seed)
                    & (df["Trial"] == 100)
                ]
                dfp.loc[:, "wall_time"] = (
                    dfp["wall_time"] * 32
                )  # Account for multi-processing
            elif strategy == "Random (10k)":
                dfp = df.loc[(df["Strategy"] == strategy) & (df["seed"] == seed)]
                dfp.loc[:, "wall_time"] = (
                    dfp["wall_time"] * 32
                )  # Account for multi-processing
            else:
                dfp = df.loc[
                    (df["Strategy"] == strategy)
                    & (df["seed"] == seed)
                    & (df["n_init"] == n_init)
                ]
            ax.scatter(
                dfp["wall_time"],
                dfp["best_obj"],
                color=common.STRATEGY_COLORS[strategy],
                **SCATTER_KWARGS,
            )

    de = de.loc[de.groupby("seed")["nit"].idxmax()]

    ax.scatter(
        de["wall_time"],
        de["obj"],
        color=common.STRATEGY_COLORS["DE"],
        **SCATTER_KWARGS,
    )

    return ax


def plot_est_dist(
    df: pd.DataFrame, de: pd.DataFrame, n_init: int = 32, ax: Optional[plt.Axes] = None
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()

    sel = (
        ((df["Strategy"] == "Random (100)") & (df["Trial"] == 100))
        | ((df["Strategy"] == "Random (10k)") & (df["Trial"] == N_RAND))
        | (
            (df["Strategy"] != "Random")
            & (df["n_init"] == n_init)
            & (df["Trial"] == 100)
        )
    )

    dfp = df.loc[sel].sort_values(
        by="Strategy", key=lambda x: x.apply(lambda y: common.SORTING_RULE.get(y, 1000))
    )

    de = de.loc[de.groupby("seed")["nit"].idxmax()]
    de_new = pd.DataFrame(columns=["Strategy", "best_obj"])
    de_new["Strategy"] = ["DE"] * len(de)
    de_new["best_obj"] = de["obj"].values
    dfp = pd.concat([dfp, de_new], axis=0, ignore_index=True)

    sns.violinplot(
        data=dfp,
        x="Strategy",
        y="best_obj",
        hue="Strategy",
        palette=common.STRATEGY_COLORS,
        fill=False,
        inner=None,
        bw_adjust=0.5,
        ax=ax,
    )
    sns.stripplot(
        data=dfp,
        x="Strategy",
        y="best_obj",
        hue="Strategy",
        palette=common.STRATEGY_COLORS,
        ax=ax,
        dodge=False,
        linewidth=0.5,
        alpha=0.25,
        size=3,
        marker="o",
    )
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    ax.set_axisbelow(True)
    ax.set_xticklabels(DIST_LABELS)
    ax.tick_params(axis="x", which="minor", bottom=False, top=False)
    return ax


def main(prepend: str = "", n_init: int = 32) -> plt.Figure:
    path = common.SWELLEX96Paths.outputs / "runs"
    data_sim = helpers.load_data(
        path, f"{prepend}sim_*/*.npz", common.SEARCH_SPACE, common.TRUE_SIM_VALUES
    )
    data_exp = helpers.load_data(
        path, f"{prepend}exp_*/*.npz", common.SEARCH_SPACE, common.TRUE_EXP_VALUES
    )

    de_sim = pd.read_csv(common.SWELLEX96Paths.outputs / "runs" / "de" / "sim_de_results.csv")
    de_exp = pd.read_csv(common.SWELLEX96Paths.outputs / "runs" / "de" / "exp_de_results.csv")

    fig = performance_plot([data_sim, data_exp], [de_sim, de_exp], n_init=n_init)
    return fig


if __name__ == "__main__":
    fig = main()
    plt.show()
