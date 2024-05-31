#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

sys.path.insert(0, str(Path(__file__).parents[1]))
from data.bo import common

plt.style.use(["science", "ieee"])


def plot_sensitivity(
    data: list[np.ndarray],
    parameters: list[dict],
):
    nrows = 7

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(3, 6),
        gridspec_kw={"hspace": 0.9, "wspace": 0.25},
    )
    print(axs.shape)
    for i, ax in enumerate(axs):

        parameter = data[0][i]

        B = parameter["value"]
        xlim = [min(parameter["space"]), max(parameter["space"])]

        ax.plot(parameter["space"], B, color="tab:blue", label="Simulated")
        ax.axvline(parameters[0][parameter["name"]], color="tab:blue", linestyle="-")
        ax.set_xlim(xlim)

        ax.tick_params(axis="y", labelcolor="tab:blue")

        ax.ticklabel_format(axis="y", style="sci", scilimits=(-6, 5))
        ax.set_xlabel(common.VARIABLES[parameter["name"]], labelpad=0)

        parameter = data[1][i]
        B = parameter["value"]

        ax2 = ax.twinx()
        ax2.plot(
            parameter["space"], B, color="tab:red", linestyle="--", label="Experimental"
        )
        ax2.axvline(parameters[1][parameter["name"]], color="tab:red", linestyle="--")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        if i == 0:
            ax.set_title("$\phi(\mathbf{m})$", y=1.1)

        if i < 3:
            ax.set_ylim(0.0, 1.0)
            ax2.set_ylim(0.0, 1.0)
        elif i == 3:
            ax.set_ylim(0.0, 0.08)
            ax.set_yticks([0.0, 0.08])
            ax2.set_ylim(0.0, 0.3)
            ax2.set_yticks([0.0, 0.3])
        elif i == 4:
            ax.set_ylim(0.0, 0.01)
            ax.set_yticks([0.0, 0.01])
            ax2.set_ylim(0.07, 0.12)
            ax2.set_yticks([0.07, 0.12])
        elif i == 5:
            ax.set_ylim(0.0, 0.00004)
            ax.set_yticks([0.0, 0.00004])
            ax.yaxis.set_ticklabels(["0.0", "$4\\times 10^{-5}$"])
            ax2.set_ylim(0.076, 0.08)
            ax2.set_yticks([0.076, 0.08])
        else:
            ax.set_ylim(0.0, 0.0008)
            ax.set_yticks([0.0, 0.0008])
            ax.yaxis.set_ticklabels(["0.0", "$8\\times 10^{-4}$"])
            ax2.set_ylim(0.076, 0.082)
            ax2.set_yticks([0.076, 0.082])

    lines1, labels1 = ax.get_legend_handles_labels()
    fig.legend(lines1, labels1, loc="upper center", ncol=1, bbox_to_anchor=(0.2, 0.925))

    lines2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(lines2, labels2, loc="upper center", ncol=1, bbox_to_anchor=(0.8, 0.925))

    return fig


def main():
    sens_sim = np.load(
        common.SWELLEX96Paths.outputs / "sensitivity_sim.npy", allow_pickle=True
    )
    sens_exp = np.load(
        common.SWELLEX96Paths.outputs / "sensitivity_exp.npy", allow_pickle=True
    )
    fig = plot_sensitivity(
        data=[sens_sim, sens_exp],
        parameters=[common.TRUE_SIM_VALUES, common.TRUE_EXP_VALUES],
    )
    return fig


if __name__ == "__main__":
    fig = main()
    plt.show()
