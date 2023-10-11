#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

sys.path.insert(0, str(Path(__file__).parents[1]))
from conf import common

sys.path.insert(0, str(Path(__file__).parents[3]))
from optimization import utils

plt.style.use(["science", "ieee"])


def plot_environment(subfiglabel: str = None):

    env = utils.load_env_from_json(common.SWELLEX96Paths.simple_environment_data)
    waterdata = env["layerdata"][0]
    z_b = waterdata["z"][-1]
    rec_z = env["rec_z"]
    c_b = env["bot_c_p"]
    src_z = common.TRUE_SRC_Z

    xlim = [1470, 1600]
    ylim = [-3, 240]

    fig, ax = plt.subplots(figsize=(3, 4), gridspec_kw={"hspace": 0.0})
    ax.axis("off")

    # SSP Model
    ax.plot(waterdata["c_p"], waterdata["z"])

    # VLA
    ax.plot([1480] * (len(rec_z) + 1), rec_z + [z_b], "k-")
    ax.scatter([1480] * len(rec_z), rec_z, marker="o", color="k", s=10)
    ax.text(1480, rec_z[0] - 5, "VLA", ha="center")

    # Source
    ax.scatter(1590, src_z, marker="o", color="k", s=10)
    ax.text(1590, src_z - 5, "Source", ha="center")

    # Surface
    ax.axhline(y=0, color="k", linestyle="-")

    # Bottom
    ax.axhline(y=z_b, color="k", linestyle="-")

    # Bottom sound speed
    ax.plot([c_b, c_b], [ylim[1], z_b], color="k", linestyle="-")
    ax.text(
        c_b + 5,
        (z_b + ylim[1]) / 2,
        "$c_b$",
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    # Bottom geoacoustic labels
    ax.text(
        xlim[0] + 50,
        (z_b + ylim[1]) / 2,
        "$\\rho_b$, $a$",
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold",
    )

    # Shaded bottom
    ax.fill_between([xlim[0], xlim[1]], z_b, ylim[1], color="lightgray")


    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.invert_yaxis()
    ax.set_ylabel("Depth [m]")


    if subfiglabel:
        fig.text(
            0.1, 0.93, subfiglabel, va="top", ha="left", fontsize=10, fontweight="bold"
        )

    return fig


def main():
    fig = plot_environment(subfiglabel="(a)")
    fig.savefig("environment.png")
    

if __name__ == "__main__":
    main()