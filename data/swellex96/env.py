#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np

from tritonoa.io.profile import read_ssp


def main(args):
    # Load CTD data
    z_data, c_data, _ = read_ssp(Path(args.ctd_path), 0, 3, header=None)
    z_data = np.append(z_data, 217).tolist()
    c_data = np.append(c_data, c_data[-1]).tolist()

    # Define environment parameters
    environment = {
        # 1. General
        "title": args.title,
        "model": args.model,
        # 2. Top medium (halfspace) - Not used
        # 3. Layered media
        "layerdata": [
            {"z": z_data, "c_p": c_data, "rho": 1},
            {"z": [217, 240], "c_p": [1572.3, 1593.0], "rho": 1.76, "a_p": 0.2},
            {"z": [240, 1040], "c_p": [1881, 3245], "rho": 2.06, "a_p": 0.06},
        ],
        # 4. Bottom medium
        "bot_opt": "A",
        "bot_c_p": 5200,
        "bot_rho": 2.66,
        "bot_a_p": 0.02,
        # 5. Speed constraints
        "clow": 0,
        "chigh": 1650,
        # 6. Receiver parameters
        "rec_z": np.linspace(94.125, 212.25, 64).tolist(),
    }

    with open(args.destination, "w") as f:
        json.dump(environment, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("ctd_path", type=str, help="Path to CTD data file.")
    parser.add_argument("destination", type=str, help="Path to destination file.")
    parser.add_argument(
        "--title", type=str, help="Title of the env file.", default="swellex96"
    )
    parser.add_argument(
        "--model", type=str, help="Model specification.", default="KRAKEN"
    )
    args = parser.parse_args()
    main(args)
