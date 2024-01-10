#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

sys.path.insert(0, str(Path(__file__).parents[1]))
from data.bo import common

plt.style.use(["science", "ieee"])


def plot_sensitivity(
    sensitivities: np.ndarray,
    title: Optional[str] = None,
    subfiglabel: Optional[str] = None,
):
    ncols = 3
    nrows = (
        len(sensitivities) // ncols
        if len(sensitivities) % ncols == 0
        else len(sensitivities) // ncols + 1
    )

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(3, 4),
        gridspec_kw={"hspace": 0.6, "wspace": 0.25},
    )
    for i, ax in enumerate(axs.reshape(-1)):
        if i >= len(sensitivities):
            ax.axis("off")
            continue
        if i % 3 != 0:
            ax.set_yticklabels([])
        if i == 0:
            ax.set_ylabel("$\phi(\mathbf{m})$ [dB]")

        parameter = sensitivities[i]
        # if parameter["name"] == "dc_p_sed":
        if "dc" in parameter["name"]:
            B = parameter["value"] + sensitivities[i - 1]["value"]
            xlim = [
                min(parameter["space"] + sensitivities[i - 1]["value"]),
                max(parameter["space"] + sensitivities[i - 1]["value"]),
            ]
        else:
            B = parameter["value"]
            xlim = [min(parameter["space"]), max(parameter["space"])]
        B = 1 - B
        B = 10 * np.log10(B / B.max())

        ax.plot(parameter["space"], B, label=parameter["name"])
        ax.axvline(common.TRUE_VALUES[parameter["name"]], color="k", linestyle="--")
        ax.set_xlim(xlim)

        if i in [0, 1, 2]:
            ax.set_ylim([-9.0, 0.5])
            ax.set_yticks([-9, -6, -3, 0])
            ax.set_yticks(np.linspace(-9, 0, 4), minor=True)
        else:
            ax.set_ylim([-3.0, 0.2])
            ax.set_yticks([-3, -2, -1, 0])
            ax.set_yticks(np.linspace(-3, 0, 4), minor=True)
        ax.set_xlabel(common.VARIABLES[parameter["name"]], labelpad=0)

    if subfiglabel:
        fig.text(
            0, 0.93, subfiglabel, va="top", ha="left", fontsize=10, fontweight="bold"
        )
    if title:
        fig.suptitle(title, y=0.93)

    return fig


def main():
    sensitivities = np.load(
        common.SWELLEX96Paths.outputs / "sensitivity_sim.npy", allow_pickle=True
    )
    fig = plot_sensitivity(sensitivities, title="Simulated data", subfiglabel="(b)")
    fig.savefig("sensitivity_sim.png")

    sensitivities = np.load(
        common.SWELLEX96Paths.outputs / "sensitivity_exp.npy", allow_pickle=True
    )
    fig = plot_sensitivity(sensitivities, title="Experimental data", subfiglabel="(c)")
    fig.savefig("sensitivity_exp.png")


if __name__ == "__main__":
    main()
