#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""This script collates data from disparate locations in the project 
directory for convenient analysis and plotting.

Usage:
    collate.py <mode> <serial>
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).parents[2]))
from conf.swellex96.optimization.common import SWELLEX96Paths

ROOT = Path.cwd() / "Data"
NO_DATA = [
    list(range(0, 6)),
    list(range(61, 73)),
    list(range(83, 91)),
    list(range(175, 186)),
    list(range(275, 281)),
    list(range(290, 296)),
]
SKIP_T = [item for sublist in NO_DATA for item in sublist]


def load_grid_parameters(path: Path):
    data = np.load(path, allow_pickle=True)
    return data["rec_r"], data["src_z"]


def load_mfp_results(ambsurf_path):
    # Load high-res MFP
    rvec, zvec = load_grid_parameters(ambsurf_path / "grid_parameters.pkl")

    timesteps = []
    ranges = []
    depths = []
    for t in range(350):
        try:
            surf = np.load(ambsurf_path / f"surface_{t:03d}.npy")
        except FileNotFoundError:
            continue
        inds = np.unravel_index(surf.argmax(), surf.shape)
        timesteps.append(t)
        depths.append(zvec[inds[0]])
        ranges.append(rvec[inds[1]])

    timesteps = np.array(timesteps)
    ranges = np.array(ranges)
    depths = np.array(depths)

    return timesteps, ranges, depths


def build_mfp_df(path: Path) -> pd.DataFrame:
    timesteps, ranges, depths = load_mfp_results(path)
    df = pd.DataFrame(data={"Time Step": timesteps, "best_rec_r": ranges, "best_src_z": depths})
    df["Time Step"] = df["Time Step"].astype(int)
    df["strategy"] = "mfp"
    return df


def load_sbl_results(path: Path):
    data = loadmat(path)["snapshotPeak"]
    data[:, 1] = data[:, 1] / 1000
    return data


def build_sbl_df(path: Path) -> pd.DataFrame:
    data = load_sbl_results(path)
    df = pd.DataFrame(data=data, columns=["Time Step", "best_rec_r", "best_src_z"])
    df["Time Step"] = df["Time Step"].astype(int) - 1
    df["strategy"] = "sbl"
    return df


def build_bogp_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        delimiter=",",
        skip_blank_lines=False,
        on_bad_lines="error",
        index_col=0,
    )
    df = df.rename(
        columns={
            "param_time_step": "Time Step",
            "param_rec_r": "Range [km]",
            "param_src_z": "Depth [m]",
        }
    )
    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace({"strategy": "mfp"}, "High-res MFP")
    df = df.replace({"strategy": "sobol"}, "Sobol")
    df = df.replace({"strategy": "grid"}, "Grid")
    df = df.replace({"strategy": "sbl"}, "SBL")
    df = df.replace({"strategy": "gpei"}, "Sobol+GP/EI")
    df = df.replace({"strategy": "qgpei"}, "Sobol+GP/qEI")
    return df


def remove_bad_rows(df) -> pd.DataFrame:
    selection = df["Time Step"].isin(SKIP_T)
    df = df.drop(df[selection].index)
    return df


def collate_data(mode: str, serial: str) -> None:
    data_path = SWELLEX96Paths.outputs / "localization" / mode / serial / "results"

    # Load high-res MFP
    df_mfp = build_mfp_df(
        SWELLEX96Paths.ambiguity_surfaces / "148-166-201-235-283-338-388_200x100"
    )

    # Load strategies
    df_bogp = build_bogp_df(data_path / "best_results.csv")

    # Load SBL results
    df_sbl = build_sbl_df(SWELLEX96Paths.sbl_data)

    # Merge and format dataframes
    df = pd.concat([df_bogp, df_sbl, df_mfp], ignore_index=True)
    df = rename_columns(df)
    df = remove_bad_rows(df)
    df = df.sort_values(by=["Time Step", "strategy"])

    fname = "collated_results.csv"
    df.to_csv(data_path / fname, index=False)
    print(f"Collated data saved to {data_path / fname}")


def get_error(df, var: str, timesteps, var_act):
    df = df.sort_values("Time Step")
    var_est = df[var].values
    est_timesteps = df["Time Step"].values
    inds = np.in1d(timesteps, est_timesteps)
    var = var_act[inds]
    error = var_est - var
    return error, est_timesteps


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=["experimental", "simulation"],
        help="Choose which optimization mode to collate.",
    )
    parser.add_argument("serial", help="Enter the data serial to collate.")
    args = parser.parse_args()

    collate_data(mode=args.mode, serial=args.serial)
