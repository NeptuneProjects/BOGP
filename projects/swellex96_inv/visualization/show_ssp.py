#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parents[1]))
from conf import common

sys.path.insert(0, str(Path(__file__).parents[3]))
from optimization import utils


def main():
    main_env = utils.load_env_from_json(common.SWELLEX96Paths.main_environment_data)

    z = main_env["layerdata"][0]["z"]
    c = main_env["layerdata"][0]["c_p"]
    plt.plot(c, z, "k-")

    z = [0.0, 5.0, 30.0, 217.0]
    c = [1522.0, 1522.0, 1490.0, 1490.0]
    plt.plot(c, z, "r-")

    plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    main()
