#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots

sys.path.insert(0, str(Path(__file__).parents[1]))
from data.bo import common, helpers

sys.path.insert(0, str(Path(__file__).parents[3]))
from optimization import utils


def plot_ssp(data: list[pd.DataFrame]) -> plt.Figure:
    env = utils.load_env_from_json(common.SWELLEX96Paths.main_environment_data)
    waterdata = env["layerdata"][0]

    df = data[1]

    # c2 = df["best_c2"]
    c3 = df["best_c3"].values
    c4 = df["best_c4"].values
    c5 = df["best_c5"].values
    c6 = df["best_c6"].values
    c7 = c6
    c1 = 1522 * np.ones_like(c3)
    c2 = c1

    z1 = np.zeros_like(c1)
    z2 = 5 * np.ones_like(c1)
    z3 = 30 * np.ones_like(c1)
    z4 = 60 * np.ones_like(c1)
    z5 = 100 * np.ones_like(c1)
    z6 = 150 * np.ones_like(c1)
    z7 = df["best_h_w"].values

    z = np.array([z1, z2, z3, z4, z5, z6, z7])
    c = np.array([c1, c2, c3, c4, c5, c6, c7])

    for i in range(c.shape[1]):
        plt.plot(c[:, i], z[:, i], "k", alpha=0.5)

    plt.plot(waterdata["c_p"], waterdata["z"], "r")
    plt.xlabel("Sound speed (m/s)")
    plt.ylabel("Depth (m)")

    plt.gca().invert_yaxis()
    plt.show()


def main():
    path = common.SWELLEX96Paths.outputs / "runs"
    data_exp = helpers.load_data(
        path, f"full_exp_ei/*.npz", common.SEARCH_SPACE, common.TRUE_VALUES
    )
    data_sim = data_exp
    plot_ssp([data_sim[data_sim["Trial"] == 500], data_exp[data_exp["Trial"] == 500]])


if __name__ == "__main__":
    main()
