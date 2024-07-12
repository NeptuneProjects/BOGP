#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from string import ascii_lowercase
import sys
from typing import Optional

import matplotlib.gridspec as gridspec
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


def plot_parameter_estimates(data: list[pd.DataFrame], parameters: list[dict]) -> plt.Figure:
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


# def plot_parameter_estimates_full(data: list[pd.DataFrame]) -> plt.Figure:
#     YLIM = [[0, 20], [0, 20]]
#     NCOLS = 7
#     NROWS = 2

#     fig = plt.figure(figsize=(8, 4))
#     gs_outer = gridspec.GridSpec(nrows=2, ncols=1, figure=fig, hspace=0.45)

#     # Tranche 1 of parameters
#     gs_inner = gridspec.GridSpecFromSubplotSpec(
#         nrows=NROWS, ncols=NCOLS, subplot_spec=gs_outer[0], wspace=0.1, hspace=0.4
#     )

#     for i in range(gs_inner.get_geometry()[0]):
#         for j in range(gs_inner.get_geometry()[1]):
#             ax = fig.add_subplot(gs_inner[i, j])

#             ax.grid(True, linestyle="dotted")

#             param_name = common.SEARCH_SPACE[j]["name"]
#             bounds = common.SEARCH_SPACE[j]["bounds"]
#             true_value = common.TRUE_VALUES[common.SEARCH_SPACE[j]["name"]]
#             if param_name == "dc_p_sed":
#                 param_name = "c_p_sed_bot"
#                 bounds = [1540, 1690]
#                 true_value = 1593.0

#             ax = plot_param_dist(data[i][f"best_{param_name}"], bounds, ax=ax)

#             ax.text(
#                 0, 1.07, f"({ascii_lowercase[NCOLS * i + j]})", transform=ax.transAxes
#             )
#             ax.axvline(
#                 true_value,
#                 color="k",
#                 linestyle="dashed",
#             )
#             ax.set_xlim(bounds)
#             ax.set_xticklabels([]) if i != 1 else None
#             ax.set_xlabel(
#                 common.VARIABLES[common.SEARCH_SPACE[j]["name"]]
#             ) if i == 1 else None
#             ax.set_ylim(YLIM[i])
#             ax.set_yticklabels([]) if j != 0 else None
#             ax.set_ylabel(f"{MODES[i]}\nCount") if j == 0 else None
#             helpers.adjust_subplotxticklabels(ax, 0, -1)

#     # Tranche 2 of parameters
#     gs_inner = gridspec.GridSpecFromSubplotSpec(
#         nrows=NROWS, ncols=NCOLS, subplot_spec=gs_outer[1], wspace=0.1, hspace=0.4
#     )

#     for i in range(gs_inner.get_geometry()[0]):
#         for j in range(gs_inner.get_geometry()[1] - 1):
#             ax = fig.add_subplot(gs_inner[i, j])

#             ax.grid(True, linestyle="dotted")

#             param_name = common.SEARCH_SPACE[j + NCOLS]["name"]
#             param_label = common.VARIABLES[common.SEARCH_SPACE[j + NCOLS]["name"]]
#             bounds = common.SEARCH_SPACE[j + NCOLS]["bounds"]
#             true_value = common.TRUE_VALUES[common.SEARCH_SPACE[j + NCOLS]["name"]]
#             if param_name == "dc_p_sed":
#                 param_name = "c_p_sed_bot"
#                 param_label = common.VARIABLES[param_name]
#                 bounds = [1540.0, 1690.0]
#                 true_value = 1593.0
#             # if param_name == "dc1":
#             #     param_name = "c2"
#             #     param_label = common.VARIABLES[param_name]
#             #     bounds = [1480, 1540]
#             #     true_value = 1499.833
#             # if param_name == "dc2":
#             #     param_name = "c3"
#             #     param_label = common.VARIABLES[param_name]
#             #     bounds = [1480, 1540]
#             #     true_value = 1492.56
#             if param_name == "dc3":
#                 param_name = "c4"
#                 param_label = common.VARIABLES[param_name]
#                 bounds = [1460.0, 1520.0]
#                 true_value = 1488.0
#                 print(data[i][f"best_{param_name}"])
#             # if param_name == "dc4":
#             #     param_name = "c5"
#             #     param_label = common.VARIABLES[param_name]
#             #     bounds = [1480, 1540]
#             #     true_value = 1489.663
#             # if param_name == "dc5":
#             #     param_name = "c6"
#             #     param_label = common.VARIABLES[param_name]
#             #     bounds = [1480, 1540]
#             #     true_value = 1488.434

#             ax = plot_param_dist(data[i][f"best_{param_name}"], bounds, ax=ax)
#             ax.text(
#                 0,
#                 1.07,
#                 f"({ascii_lowercase[NCOLS * i + j + 10]})",
#                 transform=ax.transAxes,
#             )
#             ax.axvline(
#                 true_value,
#                 color="k",
#                 linestyle="dashed",
#             )
#             ax.set_xlim(bounds)
#             ax.set_xticklabels([]) if i != 1 else None
#             ax.set_xlabel(param_label) if i == 1 else None
#             ax.set_ylim(YLIM[i])
#             ax.set_yticklabels([]) if j != 0 else None
#             ax.set_ylabel(f"{MODES[i]}\nCount") if j == 0 else None
#             helpers.adjust_subplotxticklabels(ax, 0, -1)

#     return fig


def main(strategy: str = "sobol", n_init: int = 32, prepend: str = ""):
    path = common.SWELLEX96Paths.outputs / "runs"
    data_sim = helpers.load_data(
        path, f"{prepend}sim_{strategy}/*-{n_init}_*.npz", common.SEARCH_SPACE, common.TRUE_SIM_VALUES
    )
    data_exp = helpers.load_data(
        path, f"{prepend}exp_{strategy}/*-{n_init}_*.npz", common.SEARCH_SPACE, common.TRUE_EXP_VALUES
    )
    # data_sim = data_exp
    # data_sim.to_csv("data.csv")

    if "sobol" in strategy:
        last_trial = 10000
    else:
        last_trial = 100

    if prepend in ["full_", "thermo_"]:
        print("Plotting full inversion.")
        fig = plot_parameter_estimates_full(
            [
                data_sim[data_sim["Trial"] == last_trial],
                data_exp[data_exp["Trial"] == last_trial],
            ],
        )
    else:
        print("Plotting seabed inversion.")
        fig = plot_parameter_estimates(
            [
                data_sim[data_sim["Trial"] == last_trial],
                data_exp[data_exp["Trial"] == last_trial],
            ],
            parameters=[common.TRUE_SIM_VALUES, common.TRUE_EXP_VALUES],
        )
    return fig


if __name__ == "__main__":
    fig = main(prepend="")
    fig.savefig("test_dist.png", dpi=300, bbox_inches="tight")
