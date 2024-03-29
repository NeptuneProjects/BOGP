#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_PATH = Path(
    "../data/swellex96_S5_VLA_loc_tilt/outputs/loc_tilt/experimental/serial_008/results/collated_results.csv"
    # "../data/swellex96_S5_VLA_loc_tilt/outputs/loc_tilt/simulation/serial_003/results/collated_results.csv"
)
TRUE_PATH = Path("../data/swellex96_S5_VLA_loc_tilt/gps/source_tow.csv")

KEY = {
    "rec_r": "Range [km]",
    "src_z": "Depth [m]",
    "tilt": "Apparent Tilt [deg]",
}

# SKIP_STEPS = list(range(74, 86)) + list(range(174, 181)) + list(range(189, 196))
SKIP_STEPS = list(range(37, 43)) + list(range(87, 91)) + list(range(95, 98))


def plot_results(data: pd.DataFrame, parameters: list[str]) -> plt.Figure:
    true_data = pd.read_csv(TRUE_PATH)
    r_true = true_data["Apparent Range [km]"].values
    # z_true = 60.0 * np.ones_like(r_true)
    z_true = true_data["Apparent Depth [m]"].values
    tilt_true = true_data["Apparent Tilt [deg]"].values
    true_kv = {
        "rec_r": r_true,
        "src_z": z_true,
        "tilt": tilt_true,
    }

    ylims = [
        [0.0, 6.0],
        [105.0, 15.0],
        [-4.5, 4.5],
    ]

    fig, axs = plt.subplots(4, 3, figsize=(18, 12))

    strategies = data["strategy"].unique()

    for j, parameter in enumerate(parameters):
        ax_col = axs[:, j]
        ax_col[0].set_title(KEY[parameter])
        for i, strategy in enumerate(strategies):
            ax = ax_col[i]
            df = data[data["strategy"] == strategy]
            df = df.drop(df[df["Time Step"].isin(SKIP_STEPS)].index)
            ax.scatter(
                df["Time Step"], df[f"best_{parameter}"], label=strategy, s=2.0, c="b", alpha=0.5
            )
            ax.plot(true_data.index, true_kv[parameter], "gray", alpha=0.5, label="True")
            ax.set_xlabel("Time Step")
            if j == 0:
                ax.set_ylabel(strategy)
            ax.set_ylim(ylims[j])

    fig.tight_layout()
    return fig


def main():
    data = pd.read_csv(RESULTS_PATH)
    fig = plot_results(data, ["rec_r", "src_z", "tilt"])
    plt.show()

    return


if __name__ == "__main__":
    # pass
    main()
