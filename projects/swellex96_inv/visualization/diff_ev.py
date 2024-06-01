#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parents[1]))
from data.bo import common

plt.style.use(["science", "ieee", "std-colors"])

YLIM = (0.07, 0.4)
YTICKS = [0.07, 0.1, 0.2, 0.3, 0.4]
YTICKLABELS = ["", "0.1", "0.2", "0.3", "0.4"]


def main():
    path = common.SWELLEX96Paths.outputs / "runs" / "de" / "exp_de_results.csv"
    df = pd.read_csv(path)

    nit = df[df["seed"] == 0]["nit"].values
    nfev = df[df["seed"] == 0]["nfev"].values
    time = df.pivot(index="nit", columns="seed", values="wall_time").mean(axis=1).values
    mean_ = df.pivot(index="nit", columns="seed", values="obj").mean(axis=1)
    min_ = df.pivot(index="nit", columns="seed", values="obj").min(axis=1)
    max_ = df.pivot(index="nit", columns="seed", values="obj").max(axis=1)
    
    def nit2nfev(x):
        return np.interp(
            x,
            nit,
            nfev,
        )

    def nfev2nit(x):
        return np.interp(x, nfev, nit)

    def nit2time(x):
        return np.interp(x, nit, time)

    def time2nit(x):
        return np.interp(x, time, nit)

    fig, ax = plt.subplots(figsize=(3, 1.5), nrows=1)

    ax.plot(nit, mean_, color="k")
    ax.plot(nit, min_, color="k", linestyle="-.", linewidth=0.5)
    ax.plot(nit, max_, color="k", linestyle="-.", linewidth=0.5)
    ax.set_xlim(0, max(nit))
    ax.set_yscale("log")
    ax.set_ylim(YLIM)
    ax.set_yticks(YTICKS)
    ax.set_yticklabels(YTICKLABELS)
    ax.text(-0.025, -0.04, "DE iter.", ha="right", va="top", transform=ax.transAxes)
    ax.set_ylabel("$\widehat{\phi}$", rotation=0)
    ax.grid()

    ax1 = ax.secondary_xaxis(location=-0.2, functions=(nit2nfev, nfev2nit))
    ax1.xaxis.set_major_locator(plt.FixedLocator(np.arange(0, 15000, 5000)))
    ax.text(
        -0.025, -0.24, "No. $\phi$ eval.", ha="right", va="top", transform=ax.transAxes
    )

    ax2 = ax.secondary_xaxis(location=-0.4, functions=(nit2time, time2nit))
    ax2.xaxis.set_major_locator(plt.FixedLocator([0, 100, 200, 300, 400, 500, 600]))
    ax.text(-0.025, -0.44, "Time [s]", ha="right", va="top", transform=ax.transAxes)

    return fig


if __name__ == "__main__":
    fig = main()
    plt.show()
