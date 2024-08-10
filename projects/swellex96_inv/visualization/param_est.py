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
NBINS = 50


def plot_param_dist(data: np.ndarray, bounds: list, ax: Optional[plt.Axes] = None):
    if ax is None:
        ax = plt.gca()

    sns.histplot(
        x=data, ax=ax, legend=None, element="step", bins=np.linspace(*bounds, NBINS)
    )
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    return ax


def plot_parameter_estimates(
    data: list[pd.DataFrame], parameters: list[dict]
) -> plt.Figure:
    YLIM = [[0, 80], [0, 80]]

    fig, axs = plt.subplots(
        nrows=2, ncols=7, figsize=(8, 1.5), gridspec_kw={"wspace": 0.1, "hspace": 0.4}
    )

    for i, axrow in enumerate(axs):
        df = data[i]
        for j, ax in enumerate(axrow):
            ax.grid(True, linestyle="dotted")

            param_name = common.SEARCH_SPACE[j]["name"]
            param_label = common.VARIABLES[common.SEARCH_SPACE[j]["name"]]
            bounds = common.SEARCH_SPACE[j]["bounds"]
            true_value = parameters[i][common.SEARCH_SPACE[j]["name"]]
            if param_name == "dc_p_sed":
                param_name = "c_p_sed_bot"
                param_label = common.VARIABLES[param_name]
                bounds = [1570, 1610]
                true_value = 1593.0

            ax = plot_param_dist(df[f"best_{param_name}"], bounds, ax=ax)

            ax.text(0, 1.07, f"({ascii_lowercase[7 * i + j]})", transform=ax.transAxes)
            ax.axvline(
                true_value,
                color="k",
                linestyle="dashed",
                linewidth=0.5,
            )
            ax.annotate(
                "",
                xy=(true_value, YLIM[i][1] - 3),
                xytext=(true_value, YLIM[i][1] + 10),
                arrowprops=dict(arrowstyle="simple", color="black"),
            )
            ax.set_xlim(bounds)
            ax.set_xticklabels([]) if i != 1 else None
            ax.set_xlabel(param_label) if i == 1 else None
            ax.set_ylim(YLIM[i])
            ax.set_yticklabels([]) if j != 0 else None
            ax.set_ylabel(f"Frequency\n({MODES[i]})") if j == 0 else None
            helpers.adjust_subplotxticklabels(ax, 0, -1)

    return fig


def main(strategy: str = "sobol", n_init: int = 32, prepend: str = ""):
    path = common.SWELLEX96Paths.outputs / "runs"
    data_sim = helpers.load_data(
        path,
        f"{prepend}sim_{strategy}/*-{n_init}_*.npz",
        common.SEARCH_SPACE,
        common.TRUE_SIM_VALUES,
    )
    data_exp = helpers.load_data(
        path,
        f"{prepend}exp_{strategy}/*-{n_init}_*.npz",
        common.SEARCH_SPACE,
        common.TRUE_EXP_VALUES,
    )

    if "sobol" in strategy:
        last_trial = 10000
    else:
        last_trial = 100

    return plot_parameter_estimates(
        [
            data_sim[data_sim["Trial"] == last_trial],
            data_exp[data_exp["Trial"] == last_trial],
        ],
        parameters=[common.TRUE_SIM_VALUES, common.TRUE_EXP_VALUES],
    )


if __name__ == "__main__":
    fig = main(prepend="")
    fig.savefig("test_dist.png", dpi=300, bbox_inches="tight")
