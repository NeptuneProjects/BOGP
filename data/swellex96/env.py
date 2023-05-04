#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build the acoustic environment model for SWELLEX96 and save it to JSON.
Usage:
> python data/swellex96/env.py \
    ../data/swellex96_S5_VLA/ctd/i9606.prn \
    ../data/swellex96_S5_VLA/env_models/swellex96_2.json
"""
from argparse import ArgumentParser, Namespace
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from tritonoa.io.profile import read_ssp


def build_environment(
    ctd_path: os.PathLike,
    title: str = "swellex96",
    model: str = "KRAKEN",
    tilt: Optional[float] = None,
) -> dict:
    z_data, c_data = load_ctd_data(ctd_path)
    return {
        # 1. General
        "title": title,
        "model": model,
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
        "tilt": tilt,
    }


def load_ctd_data(path: os.PathLike) -> tuple[list, list]:
    z_data, c_data, _ = read_ssp(Path(path), 0, 3, header=None)
    z_data = np.append(z_data, 217).tolist()
    c_data = np.append(c_data, c_data[-1]).tolist()
    return z_data, c_data


def load_from_json(path: os.PathLike) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_to_json(environment: dict, path: os.PathLike) -> None:
    with open(path, "w") as f:
        json.dump(environment, f, indent=4)


def main(args: Namespace) -> None:
    environment = build_environment(
        ctd_path=args.ctd_path, title=args.title, model=args.model, tilt=args.tilt
    )
    save_to_json(environment, path=args.destination)


def parse_arguments():
    parser = ArgumentParser(
        description="Builds acoustic environment model for SWELLEX96."
    )
    parser.add_argument("ctd_path", type=str, help="Path to CTD data file.")
    parser.add_argument("destination", type=str, help="Path to destination file.")
    parser.add_argument(
        "--title", type=str, help="Title of the env file.", default="swellex96"
    )
    parser.add_argument(
        "--model", type=str, help="Model specification.", default="KRAKEN"
    )
    parser.add_argument("--tilt", type=float, help="Tilt angle of the array.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_arguments())
