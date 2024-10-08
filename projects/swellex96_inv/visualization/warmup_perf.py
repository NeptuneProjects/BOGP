#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scienceplots

sys.path.insert(0, str(Path(__file__).parents[1]))
from data.bo import common, helpers

plt.style.use(["science", "ieee"])

# MODES = "Simulated"
MODES = "Experimental"
N_TRIALS = 100
YLIM = (0.05, 1.0)
YTICKS = [0.05, 0.1, 0.5, 1.0]


def plot_perf_vs_warmup(df: pd.DataFrame) -> plt.Figure:
    fig, ax1 = plt.subplots(figsize=(3.0, 1.5))

    sns.lineplot(
        data=df,
        x="n_init",
        y="best_obj",
        ax=ax1,
        err_style="bars",
        errorbar=lambda x: (x.min(), x.max()),
        err_kws={"capsize": 4.0},
        color="tab:blue",
    )
    ax1.set_ylabel("$\hat{\phi}(\mathbf{m})$", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks([8, 16, 32, 64, 96])
    ax1.tick_params(axis="x", which="minor", bottom=False, top=False)
    ax1.set_xlabel("$N_\mathrm{init}$")
    ax1.grid(True, linestyle="dotted", linewidth=0.5)

    ax2 = ax1.twinx()
    sns.lineplot(
        data=df,
        x="n_init",
        y="wall_time",
        ax=ax2,
        err_style="bars",
        errorbar=lambda x: (x.min(), x.max()),
        err_kws={"capsize": 2.0},
        color="tab:red",
    )
    ax2.set_ylabel("Wall time [s]", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    return fig


def main(strategy: str = "exp_logei"):
    path = common.SWELLEX96Paths.outputs / "runs"
    data = helpers.load_data(
        path, f"{strategy}/*.npz", common.SEARCH_SPACE, common.TRUE_EXP_VALUES
    )
    data = data.loc[data["Trial"] == N_TRIALS]
    return plot_perf_vs_warmup(data)


if __name__ == "__main__":
    main()
