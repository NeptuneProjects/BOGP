#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build the acoustic environment model for SWELLEX96 and save it to JSON.
Usage:
> python projects/swellex96_inv/data/env.py \
    ../data/swellex96_S5_VLA_inv/ctd/i9605.prn \
    ../data/swellex96_S5_VLA_inv/env_models/swellex96.json
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
    z_data = np.append(z_data, 217).tolist()
    c_data = np.append(c_data, c_data[-1]).tolist()
    return {
        # 1. General
        "title": title,
        "model": model,
        # 2. Top medium (halfspace) - Not used
        # 3. Layered media
        "layerdata": [
            {"z": z_data, "c_p": c_data, "rho": 1.0},
            {"z": [217.0, 240.0], "c_p": [1572.3, 1593.0], "rho": 1.76, "a_p": 0.2},
        ],
        # 4. Bottom medium
        "bot_opt": "A",
        "bot_c_p": 1881.0,
        "bot_rho": 2.06,
        "bot_a_p": 0.06,
        # 5. Speed constraints
        "clow": 1400.0,
        "chigh": 1800.0,
        # 6. Receiver parameters
        # "rec_z": np.linspace(94.125, 212.25, 64).tolist(),
        "rec_z": [
            94.125,
            99.755,
            105.38,
            111.0,
            116.62,
            122.25,
            127.88,
            139.12,
            144.74,
            150.38,
            155.99,
            161.62,
            167.26,
            172.88,
            178.49,
            184.12,
            189.76,
            195.38,
            200.99,
            206.62,
            212.25,
        ],
        "tilt": tilt,
        "z_pivot": 217.0,
    }


def format_ssp(z_data: np.ndarray, c_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z1, c1 = z_data[0], c_data[0]

    # To be estimated:
    ind2 = np.argmin(np.abs(z_data - 5))
    z2, c2 = z_data[ind2], c_data[ind2]
    ind3 = np.argmin(np.abs(z_data - 15))
    z3, c3 = z_data[ind3], c_data[ind3]
    ind4 = np.argmin(np.abs(z_data - 30))
    z4, c4 = z_data[ind4], c_data[ind4]
    ind5 = np.argmin(np.abs(z_data - 60))
    z5, c5 = z_data[ind5], c_data[ind5]
    ind6 = np.argmin(np.abs(z_data - 100))
    z6, c6 = z_data[ind6], c_data[ind6]
    return np.array([z1, z2, z3, z4, z5, z6]), np.array([c1, c2, c3, c4, c5, c6])

    # ind2 = np.argmin(np.abs(z_data - 10))
    # z2, c2 = z_data[ind2], c_data[ind2]
    # ind3 = np.argmin(np.abs(z_data - 30))
    # z3, c3 = z_data[ind3], c_data[ind3]
    # ind4 = np.argmin(np.abs(z_data - 60))
    # z4, c4 = z_data[ind4], c_data[ind4]
    # ind5 = np.argmin(np.abs(z_data - 100))
    # z5, c5 = z_data[ind5], c_data[ind5]

    # return np.array([z1, z2, z3, z4, z5]), np.array([c1, c2, c3, c4, c5])


def build_simple_environment(
    ctd_path: os.PathLike,
    title: str = "swellex96",
    model: str = "KRAKEN",
    tilt: Optional[float] = None,
) -> dict:
    z_data, c_data = load_ctd_data(ctd_path)
    # slice_every = 60
    # if slice_every:
    #     z_data = z_data[::slice_every]
    #     c_data = c_data[::slice_every]
    z_data, c_data = format_ssp(z_data, c_data)
    print(np.diff(c_data))
    z_data = np.append(z_data, 217).tolist()
    c_data = np.append(c_data, c_data[-1]).tolist()
    return {
        # 1. General
        "title": title,
        "model": model,
        # 2. Top medium (halfspace) - Not used
        # 3. Layered media
        "layerdata": [
            # {"z": [0.0, 35.0, 100.0, 217.0], "c_p": [1522.0, 1493.4, 1488.4, 1488.9], "rho": 1.0},
            {"z": z_data, "c_p": c_data, "rho": 1.0},
            # {"z": [217.0, 240.0], "c_p": [1572.3, 1593.0], "rho": 1.76, "a_p": 0.2},
        ],
        # 4. Bottom medium
        "bot_opt": "A",
        "bot_c_p": 1572.3,
        "bot_rho": 1.76,
        "bot_a_p": 0.2,
        # "bot_c_p": 1881.0,
        # "bot_rho": 2.06,
        # "bot_a_p": 0.06,
        # 5. Speed constraints
        "clow": 1400.0,
        "chigh": 1800.0,
        # 6. Receiver parameters
        # "rec_z": np.linspace(94.125, 212.25, 64).tolist(),
        "rec_z": [
            94.125,
            99.755,
            105.38,
            111.0,
            116.62,
            122.25,
            127.88,
            139.12,
            144.74,
            150.38,
            155.99,
            161.62,
            167.26,
            172.88,
            178.49,
            184.12,
            189.76,
            195.38,
            200.99,
            206.62,
            212.25,
        ],
        "tilt": tilt,
        "z_pivot": 217.0,
    }


def load_ctd_data(path: os.PathLike) -> tuple[list, list]:
    z_data, c_data, _ = read_ssp(Path(path), 0, 3, header=None)
    return z_data, c_data


def load_from_json(path: os.PathLike) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def save_to_json(environment: dict, path: os.PathLike) -> None:
    with open(path, "w") as f:
        json.dump(environment, f, indent=4)


def main(args: Namespace) -> None:
    main_environment = build_environment(
        ctd_path=args.ctd_path, title=args.title, model=args.model, tilt=args.tilt
    )
    simple_environment = build_simple_environment(
        ctd_path=args.ctd_path, title=args.title, model=args.model, tilt=args.tilt
    )

    save_to_json(main_environment, path=args.destination / "main_env.json")
    save_to_json(simple_environment, path=args.destination / "simple_env.json")


def parse_arguments():
    parser = ArgumentParser(
        description="Builds acoustic environment model for SWELLEX96."
    )
    parser.add_argument(
        "--ctd_path",
        type=str,
        help="Path to CTD data file.",
        default="../data/swellex96_S5_VLA_inv/ctd/i9605.prn",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        help="Path to destination file.",
        default="../data/swellex96_S5_VLA_inv/env_models",
    )
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
