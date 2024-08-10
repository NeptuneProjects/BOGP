#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
MP_MULT = 32
T0 = 0.05


def time_projection_plot(
    df: pd.DataFrame, de: pd.DataFrame, n_init: int = 32
) -> plt.Figure:
    df = helpers.split_random_results(df, 100)
    df = get_time_constants(df, de, n_init)

    T1 = np.logspace(-2, 2, 100)
    np.logspace

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))

    for strategy in df["Strategy"].unique():
        time_model = get_time_model(strategy)
        dfp = df.loc[df["Strategy"] == strategy]
        ax.plot(
            T1,
            time_model(T1, dfp["wall_time"].values[0]),
            color=common.STRATEGY_COLORS[strategy],
            label=strategy,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, linestyle="-", which="major", linewidth=0.5)
    ax.grid(True, linestyle=":", which="minor", linewidth=0.25)
    ax.set_xlim(0.01, 100)
    ax.set_ylim(1e0, 3e6)
    ax.tick_params(axis="both", which="major", labelsize=10)
    ax.set_xlabel("Objective Function Duration [s]", size=12)
    ax.set_ylabel("Projected Wall Time [s]", size=12)
    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[3], handles[4], handles[5], handles[2], handles[0], handles[1]]
    labels = [labels[3], labels[4], labels[5], labels[2], labels[0], labels[1]]
    ax.legend(
        handles,
        labels,
        prop={"size": 8},
        ncol=2,
        loc="upper left",
        fancybox=False,
        frameon=True,
        framealpha=1.0,
        title="Strategy",
        title_fontsize=10,
    )
    return fig


def get_time_model(strategy: str) -> callable:
    if strategy in ["BO-UCB", "BO-EI", "BO-LogEI"]:
        return bo_time_model
    return direct_time_model


def bo_time_model(T1: np.ndarray, wall_time: float) -> np.ndarray:
    return wall_time + 100 * (T1 - T0)


def direct_time_model(T1: np.ndarray, wall_time: float) -> np.ndarray:
    return wall_time * T1 / T0


def get_time_constants(
    df: pd.DataFrame, de: pd.DataFrame, n_init: int = 64
) -> pd.DataFrame:
    sel = (
        ((df["Strategy"] == "Random (100)") & (df["Trial"] == 100))
        | ((df["Strategy"] == "Random (10k)") & (df["Trial"] == N_RAND))
        | (
            (df["Strategy"] != "Random")
            & (df["n_init"] == n_init)
            & (df["Trial"] == 100)
        )
    )
    df = (
        df.loc[sel]
        .drop(columns=["Trial", "seed"])
        .groupby(["Strategy"])
        .mean()
        .reset_index()
    )
    df = df.drop(columns=df.columns.difference(["Strategy", "wall_time"]), axis=1)
    subsel = (df["Strategy"] == "Random (100)") | (df["Strategy"] == "Random (10k)")
    df.loc[subsel, "wall_time"] = df.loc[subsel, "wall_time"] * MP_MULT

    de = de.loc[de.groupby("seed")["nit"].idxmax()]
    de["Strategy"] = "DE"
    de = (
        de.drop(columns=de.columns.difference(["Strategy", "wall_time"]), axis=1)
        .groupby(["Strategy"])
        .mean()
        .reset_index()
    )
    return pd.concat([df, de], ignore_index=True)


def main(prepend: str = "", n_init: int = 64) -> plt.Figure:
    path = common.SWELLEX96Paths.outputs / "runs"
    df_exp = helpers.load_data(
        path, f"{prepend}exp_*/*.npz", common.SEARCH_SPACE, common.TRUE_EXP_VALUES
    )
    de_exp = pd.read_csv(
        common.SWELLEX96Paths.outputs / "runs" / "de" / "exp_de_results.csv"
    )

    fig = time_projection_plot(df_exp, de_exp, n_init=n_init)
    return fig


if __name__ == "__main__":
    fig = main()
    plt.show()
