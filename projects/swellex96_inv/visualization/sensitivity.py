#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from conf import common

TRUE_VALUES = {
    "rec_r": 4.15,
    "src_z": 66.3,
    "h_w": 217.0,
    "h_s": 23.0,
    "c_s": 1572.3,
    "dcdz_s": 0.9,
    "tilt": 0.4,
}

def main():
    
    sensitivities = np.load(common.SWELLEX96Paths.outputs / "sensitivity.npy", allow_pickle=True)

    fig, axs = plt.subplots(nrows=len(sensitivities), ncols=1, figsize=(6, 12))
    for i, parameter in enumerate(sensitivities):
        B = parameter["value"]
        B = 1 - B
        B = 10 * np.log10(B / B.max())
        # print(B.min())

        ax = axs[i]
        ax.plot(parameter["space"], B, label=parameter["name"])
        ax.axvline(TRUE_VALUES[parameter["name"]], color="k", linestyle="--")
        ax.set_ylim([-6.0, 0.1])
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
