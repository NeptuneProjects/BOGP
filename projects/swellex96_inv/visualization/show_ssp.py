#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

sys.path.insert(0, str(Path(__file__).parents[1]))
from conf import common

sys.path.insert(0, str(Path(__file__).parents[3]))
from optimization import utils


def main():
    main_env = utils.load_env_from_json(common.SWELLEX96Paths.main_environment_data)
    simple_env = utils.load_env_from_json(common.SWELLEX96Paths.simple_environment_data)

    z = main_env["layerdata"][0]["z"]
    c = main_env["layerdata"][0]["c_p"]
    plt.plot(c, z, "k-")

    z = simple_env["layerdata"][0]["z"]
    c = simple_env["layerdata"][0]["c_p"]
    plt.plot(c, z, "o")

    print(z)
    cs = CubicSpline(z, c)
    zs = np.linspace(z[1], z[-2], 21).tolist()    
    z_new = [z[0]] + zs + [z[-1]]
    c_new = [c[0]] + cs(zs).tolist() + [c[-1]]
    plt.plot(c_new, z_new)

    # cs = CubicSpline(z, c, bc_type="clamped")
    # zs = np.linspace(z[1], z[-2], 21).tolist()    
    # z_new = [z[0]] + zs + [z[-1]]
    # c_new = [c[0]] + cs(zs).tolist() + [c[-1]]
    # plt.plot(c_new, z_new)
    # cs = CubicSpline(z, c, bc_type="natural")
    # zs = np.linspace(z[1], z[-2], 21).tolist()    
    # z_new = [z[0]] + zs + [z[-1]]
    # c_new = [c[0]] + cs(zs).tolist() + [c[-1]]
    # plt.plot(c_new, z_new)


    plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    main()
