#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

sys.path.insert(0, str(Path(__file__).parents[1]))
from data.bo import common

sys.path.insert(0, str(Path(__file__).parents[3]))
from optimization import utils

plt.style.use(["science", "ieee"])

SHADED_KWARGS = {
    "color": "lightgray",
    "linewidth": 0,
    "zorder": -1,
    "rasterized": True,
}
BOUNDARY_KWARGS = {
    "color": "k",
    "linewidth": 0.5,
    "linestyle": "-",
}
TEXT_KWARGS = {
    "ha": "center",
    "va": "center",
    "fontsize": 7,
}
CLABEL_XLOC = -0.1
GLABEL_XLOC = 0.6
VLA_XLOC = 0.1
SRC_XLOC = 0.85


def plot_environment(subfiglabel: str = None):

    env = utils.load_env_from_json(common.SWELLEX96Paths.main_environment_data_sim)
    waterdata = env["layerdata"][0]

    boundary_locs = [0, 217, 240, 270, 300]
    boundary_tick_labels = [0, 50, 100, 150, 200, 217, 240, 1040]
    ytick_locs = [0, 50, 100, 150, 200, 217, 240, 270]
    ytick_minor_locs = np.arange(0, 218, 10).tolist()

    z_b = waterdata["z"][-1]
    rec_z = env["rec_z"]
    src_z = common.TRUE_SRC_Z

    ylim = [boundary_locs[0], boundary_locs[-1]]

    fig, axs = plt.subplots(
        ncols=2, figsize=(3, 4), gridspec_kw={"width_ratios": [0.3, 0.7], "wspace": 0.0}
    )

    # === SSP Model ===
    ax = axs[0]
    xlim = [1475, 1525]
    xtick_locs = [1480, 1500, 1520]

    # Water SSP
    ax.plot(waterdata["c_p"], waterdata["z"])

    # Sediment 1
    ax.fill_between(
        [xlim[0], xlim[1]],
        boundary_locs[1],
        boundary_locs[2],
        alpha=0.25,
        **SHADED_KWARGS,
    )

    # Sediment 2
    ax.fill_between(
        [xlim[0], xlim[1]],
        boundary_locs[2],
        boundary_locs[3],
        alpha=0.5,
        **SHADED_KWARGS,
    )

    # Bottom
    ax.fill_between(
        [xlim[0], xlim[1]],
        boundary_locs[3],
        boundary_locs[4],
        alpha=0.75,
        **SHADED_KWARGS,
    )

    ax.hlines(y=[boundary_locs[1:-1]], xmin=xlim[0], xmax=xlim[1], **BOUNDARY_KWARGS)

    ax.spines["right"].set_visible(False)
    ax.xaxis.set_ticks_position("top")
    ax.set_xticks(xtick_locs)
    ax.xaxis.get_ticklabels()[1].set_visible(False)
    ax.set_xticks(np.arange(xlim[0], xlim[1] + 5, 5), minor=True)

    ax.set_yticks(ytick_locs)
    ax.set_yticks(ytick_minor_locs, minor=True)
    ax.set_yticklabels(boundary_tick_labels)

    ax.xaxis.set_label_position("top")
    ax.yaxis.set_ticks_position("left")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("Sound speed [m/s]")
    ax.set_ylabel("Depth [m]", labelpad=0.0)

    ax.invert_yaxis()

    # === Geometric & Geoacoustic Model ===
    ax = axs[1]
    xlim = [0.0, 1.0]
    ax.spines["left"].set_visible(False)

    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])

    # VLA
    ax.plot([VLA_XLOC] * (len(rec_z) + 1), rec_z + [z_b], "k-")
    ax.scatter([VLA_XLOC] * len(rec_z), rec_z, marker="o", color="k", s=10)
    ax.text(VLA_XLOC, rec_z[0] - 5, "VLA", ha="center")

    # Source
    ax.scatter(SRC_XLOC, src_z, marker="o", color="k", s=10)
    ax.text(SRC_XLOC, src_z - 10, "Source", ha="center")

    # Range Arrow
    arr = mpatches.FancyArrowPatch(
        (SRC_XLOC, 70),
        (VLA_XLOC, 70),
        arrowstyle="|-|,widthA=0.4,widthB=0.4",
        mutation_scale=10,
        linewidth=0.5,
        color="k",
    )
    ax.add_patch(arr)
    ax.text(
        (SRC_XLOC + VLA_XLOC) / 2,
        63,
        "$r_\mathrm{src}=1.07$ km",
        ha="center",
        va="center",
        fontsize=7,
    )
    ax.text(
        (SRC_XLOC + VLA_XLOC) / 2,
        77,
        "$z_\mathrm{src}=60$ m",
        ha="center",
        va="center",
        fontsize=7,
    )

    ax.patch.set_alpha(0)

    # Sediment 1
    ax.fill_between(
        [xlim[0], xlim[1]],
        boundary_locs[1],
        boundary_locs[2],
        alpha=0.25,
        **SHADED_KWARGS,
    )
    ax.text(
        CLABEL_XLOC,
        boundary_locs[1] + 6,
        "$c_{s,t} = 1572$ m/s",
        **TEXT_KWARGS,
    )
    ax.text(
        CLABEL_XLOC,
        boundary_locs[2] - 6,
        "$c_{s,b} = 1593$ m/s",
        **TEXT_KWARGS,
    )
    ax.text(
        GLABEL_XLOC,
        boundary_locs[1] + 6,
        "$\\rho_{s} = 1.76$ g/cm\\textsuperscript{3}",
        **TEXT_KWARGS,
    )
    ax.text(
        GLABEL_XLOC,
        boundary_locs[2] - 6,
        "$a_{s} = 0.2$ dB/km·Hz",
        **TEXT_KWARGS,
    )

    # Sediment 2
    ax.fill_between(
        [xlim[0], xlim[1]],
        boundary_locs[2],
        boundary_locs[3],
        alpha=0.5,
        **SHADED_KWARGS,
    )
    ax.text(
        CLABEL_XLOC,
        boundary_locs[2] + 6,
        "$c_{m,t} = 1881$ m/s",
        **TEXT_KWARGS,
    )
    ax.text(
        CLABEL_XLOC,
        boundary_locs[3] - 6,
        "$c_{m,b} = 3245$ m/s",
        **TEXT_KWARGS,
    )
    ax.text(
        GLABEL_XLOC,
        boundary_locs[2] + 9,
        "$\\rho_{m} = 2.06$ g/cm\\textsuperscript{3}",
        **TEXT_KWARGS,
    )
    ax.text(
        GLABEL_XLOC,
        boundary_locs[3] - 9,
        "$a_{m} = 0.06$ dB/km·Hz",
        **TEXT_KWARGS,
    )

    # Bottom
    ax.fill_between(
        [xlim[0], xlim[1]],
        boundary_locs[3],
        boundary_locs[4],
        alpha=0.75,
        **SHADED_KWARGS,
    )
    ax.text(
        CLABEL_XLOC,
        boundary_locs[3] + 15,
        "$c_b = 5200$ m/s",
        **TEXT_KWARGS,
    )
    ax.text(
        GLABEL_XLOC,
        boundary_locs[3] + 9,
        "$\\rho_{b} = 2.66$ g/cm\\textsuperscript{3}",
        **TEXT_KWARGS,
    )
    ax.text(
        GLABEL_XLOC,
        boundary_locs[3] + 21,
        "$a_{b} = 0.02$ dB/km·Hz",
        **TEXT_KWARGS,
    )

    ax.hlines(y=[boundary_locs[1:-1]], xmin=xlim[0], xmax=xlim[1], **BOUNDARY_KWARGS)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()

    if subfiglabel:
        fig.text(
            -0.03,
            0.96,
            subfiglabel,
            va="top",
            ha="left",
            fontsize=10,
            fontweight="bold",
        )

    return fig


def main():
    fig = plot_environment(subfiglabel=None)
    return fig


if __name__ == "__main__":
    fig = main()
    plt.show()
