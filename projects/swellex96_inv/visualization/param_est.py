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
YLIM = [[0, 20], [0, 25]]


def plot_param_dist(data: np.ndarray, bounds: list, ax: Optional[plt.Axes] = None):
    if ax is None:
        ax = plt.gca()

    sns.histplot(x=data, ax=ax, legend=None, element="step", bins=np.linspace(*bounds, NBINS))
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    return ax


def plot_parameter_estimates(data: list[pd.DataFrame]) -> plt.Figure:
    fig, axs = plt.subplots(
        nrows=2, ncols=7, figsize=(8, 1.5), gridspec_kw={"wspace": 0.1, "hspace": 0.4}
    )

    for i, axrow in enumerate(axs):
        df = data[i]
        for j, ax in enumerate(axrow):
            ax.grid(True, linestyle="dotted")
            
            param_name = common.SEARCH_SPACE[j]["name"]
            bounds = common.SEARCH_SPACE[j]["bounds"]
            true_value = common.TRUE_VALUES[common.SEARCH_SPACE[j]["name"]]
            if param_name == "dc_p_sed":
                param_name = "c_p_sed_bot"
                bounds = [1540, 1690]
                true_value = 1593.0
            
            ax = plot_param_dist(df[f"best_{param_name}"], bounds, ax=ax)
            
            ax.text(0, 1.07, f"({ascii_lowercase[7 * i + j]})", transform=ax.transAxes)
            ax.axvline(
                true_value,
                color="k",
                linestyle="dashed",
            )
            ax.set_xlim(bounds)
            ax.set_xticklabels([]) if i != 1 else None
            ax.set_xlabel(
                common.VARIABLES[common.SEARCH_SPACE[j]["name"]]
            ) if i == 1 else None
            ax.set_ylim(YLIM[i])
            ax.set_yticklabels([]) if j != 0 else None
            ax.set_ylabel(f"{MODES[i]}\nCount") if j == 0 else None
            helpers.adjust_subplotxticklabels(ax, 0, -1)

    return fig


def main():
    path = common.SWELLEX96Paths.outputs / "runs"
    data_sim = helpers.load_data(
        path, "sim_ei/*.npz", common.SEARCH_SPACE, common.TRUE_VALUES
    )
    data_exp = helpers.load_data(
        path, "exp_ei/*.npz", common.SEARCH_SPACE, common.TRUE_VALUES
    )
    data_sim.to_csv("data.csv")
    fig = plot_parameter_estimates(
        [data_sim[data_sim["Trial"] == 500], data_exp[data_exp["Trial"] == 500]]
    )
    return fig


if __name__ == "__main__":
    main()
