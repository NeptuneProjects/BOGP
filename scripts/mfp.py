#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path.cwd() / "Source"))
from BOGP import acoustics
from tritonoa.io import read_ssp

ROOT = Path.home() / "Research" / "Projects" / "BOGP"
ranges = [0.5, 3.0, 6.0, 10.0]
# ranges = [1.0, 3.0, 5.0, 7.0]
depth_true = 62
freq = 201


def main(path, dr, dz, nr, nz):
    # Load CTD data
    z_data, c_data, _ = read_ssp(
        ROOT / "Data" / "SWELLEX96" / "CTD" / "i9606.prn", 0, 3, header=None
    )
    z_data = np.append(z_data, 217)
    c_data = np.append(c_data, c_data[-1])

    fixed_parameters = {
        # General
        "title": "SWELLEX96_SIM",
        "tmpdir": ".",
        "model": "KRAKENC",
        # Top medium
        # Layered media
        "layerdata": [
            {"z": z_data, "c_p": c_data, "rho": 1},
            {"z": [217, 240], "c_p": [1572.37, 1593.02], "rho": 1.8, "a_p": 0.3},
            {"z": [240, 1040], "c_p": [1881, 3245.8], "rho": 2.1, "a_p": 0.09},
        ],
        # Bottom medium
        "bot_opt": "A",
        "bot_c_p": 5200,
        "bot_rho": 2.7,
        "bot_a_p": 0.03,
        # Speed constraints
        "clow": 0,
        "chigh": 1600,
        # Receiver parameters
        "rec_z": np.linspace(94.125, 212.25, 64),
        # "rec_r": range_true,
        "snr": 20,
        # Source parameters
        # "src_z": depth_true,
        "freq": freq,
    }
    search_parameters = [
        {"name": "rec_r", "bounds": [0.001, 10.0]},
        {"name": "src_z", "bounds": [0.5, 200.0]},
    ]

    for r in ranges:
        fixed_parameters["rec_r"] = r
        fixed_parameters["src_z"] = depth_true
        parameters = {
            "fixed_parameters": fixed_parameters,
            "search_parameters": search_parameters,
        }
        B, rvec, zvec = acoustics.run_mfp(parameters, dr=dr, dz=dz, nr=nr, nz=nz)
        np.savez(
            Path(path) / f"{freq}Hz_{depth_true}m_{r}km",
            # ROOT
            # / "Data"
            # / "SWELLEX96"
            # / "ambiguity_surfaces"
            # / f"{freq}Hz_{depth_true}m_{r}km",
            B=B,
            rvec=rvec,
            zvec=zvec,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--dr", type=float)
    parser.add_argument("--dz", type=float)
    parser.add_argument("--nr", type=int)
    parser.add_argument("--nz", type=int)
    args = parser.parse_args()
    main(**vars(args))
