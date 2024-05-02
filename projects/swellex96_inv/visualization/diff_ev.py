#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

sys.path.insert(0, str(Path(__file__).parents[1]))
from data.bo import common

plt.style.use(["science", "ieee", "std-colors"])


def load_de_results(path: Path) -> tuple:
    data = np.load(path)
    return data["nit"], data["nfev"], data["elapsed_time"], data["fun"]


def main():
    path = common.SWELLEX96Paths.outputs / "de_results_exp.npz"
    nit, nfev, elapsed_time, fun = load_de_results(path)

    def nit2nfev(x):
        return np.interp(
            x,
            nit,
            nfev,
        )

    def nfev2nit(x):
        return np.interp(x, nfev, nit)

    def nit2time(x):
        return np.interp(x, nit, elapsed_time)

    def time2nit(x):
        return np.interp(x, elapsed_time, nit)

    fig, ax = plt.subplots(figsize=(3, 1.5), nrows=1)

    ax0 = ax
    ax0.plot(nit, fun, label="$\widehat{\phi}$")
    ax0.text(-0.025, -0.04, "DE iter.", ha="right", va="top", transform=ax0.transAxes)
    ax0.set_ylabel("$\widehat{\phi}$", rotation=0)
    ax0.set_xlim(0, 40)
    ax0.grid()

    ax1 = ax0.secondary_xaxis(location=-0.2, functions=(nit2nfev, nfev2nit))
    ax1.xaxis.set_major_locator(plt.FixedLocator(np.arange(0, 25000, 5000)))
    ax0.text(
        -0.025, -0.24, "No. $\phi$ eval.", ha="right", va="top", transform=ax0.transAxes
    )

    ax2 = ax0.secondary_xaxis(location=-0.4, functions=(nit2time, time2nit))
    ax2.xaxis.set_major_locator(plt.FixedLocator([0, 200, 400, 600, 800]))
    ax0.text(-0.025, -0.44, "Time [s]", ha="right", va="top", transform=ax0.transAxes)

    return fig


if __name__ == "__main__":
    fig = main()
    plt.show()
