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
from scipy.interpolate import Akima1DInterpolator
import seaborn as sns
import scienceplots

sys.path.insert(0, str(Path(__file__).parents[1]))
from data.bo import common, helpers

sys.path.insert(0, str(Path(__file__).parents[3]))
from optimization import utils


def plot_ssp(data: list[pd.DataFrame]) -> plt.Figure:
    env = utils.load_env_from_json(common.SWELLEX96Paths.main_environment_data)
    waterdata = env["layerdata"][0]

    df = data[1]

    c1 = 1522
    # c2 = df["best_c2"]
    # c3 = df["best_c3"]
    # c4 = df["best_c4"]
    # c5 = df["best_c5"]
    # c6 = df["best_c6"]
    # c7 = c6
    # c1 = np.ones_like(c7) * c1
    # c = np.array([c1, c2, c3, c4, c5, c6, c7])
    # z = np.array([0, 20, 40, 60, 80, 100, 217])

    c3 = df["best_c3"]
    # c4 = df["best_c4"]
    c4 = c3
    c1 = np.ones_like(c3) * c1
    c2 = c1
    c = np.array([c1, c2, c3, c4])

    z3 = df["best_z3"]
    z4 = df["best_h_w"]
    z = np.array([np.zeros_like(z3), np.ones_like(z3) * 5, z3, np.ones_like(z3) * z4])

    for i in range(c.shape[1]):
        # zs = np.linspace(z[1], z[-2], 50).tolist()
        # cs = Akima1DInterpolator(z, c[:, i])

        # zq = [z[0]] + zs + [z[-1]]
        # cq = [c[0, i]] + cs(zs).tolist() + [c[-1, i]]
        # plt.plot(cq, zq, "k")

        plt.plot(c[:, i], z[:, i], "k", alpha=0.5)

    plt.plot(waterdata["c_p"], waterdata["z"], "r")
    plt.xlabel("Sound speed (m/s)")
    plt.ylabel("Depth (m)")

    plt.gca().invert_yaxis()
    plt.show()


def main():
    path = common.SWELLEX96Paths.outputs / "runs"
    # data_sim = helpers.load_data(
    #     path, f"thermo_sim_ei/*.npz", common.SEARCH_SPACE, common.TRUE_VALUES
    # )
    data_exp = helpers.load_data(
        path, f"thermo_exp_ei/*.npz", common.SEARCH_SPACE, common.TRUE_VALUES
    )
    data_sim = data_exp
    plot_ssp([data_sim[data_sim["Trial"] == 500], data_exp[data_exp["Trial"] == 500]])


if __name__ == "__main__":
    main()
