#!/usr/bin/env python3
# Copyright 2023 by William Jenkins
# Scripps Institution of Oceanography
# University of California San Diego
#
# This source code is licensed under the MIT license found in the LICENSE
# file in the root directory of this source tree.

"""This script collates data from disparate locations in the project 
directory for convenient analysis and plotting.
"""

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path.cwd() / "Data"


def load_mfp_results(ambsurf_path):
    # Load high-res MFP
    zvec = np.linspace(1, 200, 100)
    rvec = np.linspace(10e-3, 10, 500)

    timesteps = []
    ranges = []
    depths = []
    for t in range(350):
        try:
            surf = np.load(ambsurf_path / f"ambsurf_mf_t={t + 1:03d}.npy")
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


def collate_experimental(path):
    AMBSURF_PATH = (
        ROOT
        / "SWELLEX96"
        / "VLA"
        / "selected"
        / "multifreq"
        / "148.0-166.0-201.0-235.0-283.0-338.0-388.0"
    )
    NO_DATA = [
        list(range(73, 85)),
        list(range(95, 103)),
        list(range(187, 199)),
        list(range(287, 294)),
        list(range(302, 309)),
    ]
    SKIP_T = (
        [49, 72, 94, 186, 286, 301]
        + NO_DATA[0]
        + NO_DATA[1]
        + NO_DATA[2]
        + NO_DATA[3]
        + NO_DATA[4]
    )

    print(f"Reading data from {path}")

    # Load high-res MFP
    timesteps, ranges, depths = load_mfp_results(AMBSURF_PATH)

    df_mfp = pd.DataFrame(
        data={"Time Step": timesteps, "rec_r": ranges, "src_z": depths}
    )
    df_mfp["Time Step"] = df_mfp["Time Step"].astype(int)
    selection = df_mfp["Time Step"].isin(SKIP_T)
    df_mfp = df_mfp.drop(df_mfp[selection].index)
    df_mfp["strategy"] = "High-res MFP"

    # Load strategies
    fname = path / "results" / "best_results.csv"

    df = pd.read_csv(
        fname,
        delimiter=",",
        skip_blank_lines=False,
        on_bad_lines="error",
    )
    df["scenario"] = (
        df["scenario"]
        .apply(lambda x: ast.literal_eval(x))
        .apply(lambda x: list(x.values())[0])
    )
    df = df.rename(columns={"scenario": "Time Step"})
    df = df.loc[:, ~df.columns.str.match("Unnamed")]
    df = df.rename(columns={"scenario": "Time Step"})
    df = df.replace({"strategy": "sequential_qpi1"}, "PI")
    df = df.replace({"strategy": "sequential_qei1"}, "EI")
    df = df.replace({"strategy": "greedy_batch_qei5"}, "qEI")
    df = df.replace({"strategy": "lhs"}, "LHS")
    df = df.replace({"strategy": "random"}, "Rand")
    df = df.replace({"strategy": "sobol"}, "Sobol")
    df = df.replace({"strategy": "grid"}, "Grid")
    df["trial_index"] = df["trial_index"] + 1
    df["trial_index"] = df["trial_index"].astype(int)
    selection = df["Time Step"].isin(SKIP_T)
    df = df.drop(df[selection].index)

    # Load SBL results

    sbl_data = loadmat(
        ROOT / "SWELLEX96" / "VLA" / "selected" / "SBL" / "results_constrained.mat"
    )["snapshotPeak"]
    sbl_data[:, 1:] = np.flipud(sbl_data[:, 1:])
    sbl_data[:, 1] = sbl_data[:, 1] / 1000
    df_sbl = pd.DataFrame(data=sbl_data, columns=["Time Step", "rec_r", "src_z"])
    df_sbl["Time Step"] = df_sbl["Time Step"].astype(int) - 1
    selection = df_sbl["Time Step"].isin(SKIP_T)
    df_sbl = df_sbl.drop(df_sbl[selection].index)
    df_sbl["strategy"] = "SBL"

    df = pd.concat([df, df_sbl, df_mfp], ignore_index=True)

    savepath = path / "results" / "collated.csv"
    df.to_csv(savepath)
    print(f"Collated data saved to {savepath}")


def collate_simulation(path):
    print(f"Reading data from {path}")
    df = pd.read_csv(path / "results" / "aggregated_results.csv")
    df = df.loc[:, ~df.columns.str.match("Unnamed")]

    df["range"] = (
        df["scenario"]
        .apply(lambda x: ast.literal_eval(x))
        .apply(lambda x: list(x.values())[0])
    )
    df["depth"] = (
        df["scenario"]
        .apply(lambda x: ast.literal_eval(x))
        .apply(lambda x: list(x.values())[1])
    )
    df["best_range"] = (
        df["best_parameters"]
        .apply(lambda x: ast.literal_eval(x))
        .apply(lambda x: list(x.values())[0])
    )
    df["best_range_error"] = np.abs(df["best_range"] - df["range"])

    if path.parents[1].name == "localization":
        df["best_depth"] = (
            df["best_parameters"]
            .apply(lambda x: ast.literal_eval(x))
            .apply(lambda x: list(x.values())[1])
        )
        df["best_depth_error"] = np.abs(df["best_depth"] - df["depth"])

    df = df.replace({"strategy": "sequential_qpi1"}, "PI")
    df = df.replace({"strategy": "sequential_qei1"}, "EI")
    df = df.replace({"strategy": "greedy_batch_qei5"}, "qEI")
    df = df.replace({"strategy": "lhs"}, "LHS")
    df = df.replace({"strategy": "random"}, "Rand")
    df = df.replace({"strategy": "sobol"}, "Sobol")
    df = df.replace({"strategy": "grid"}, "Grid")
    df["trial_index"] = df["trial_index"] + 1

    savepath = path / "results" / "collated.csv"
    df.to_csv(savepath)
    print(f"Collated data saved to {savepath}")


def format_args(args):
    if args.optim == "r":
        args.optim = "range_estimation"
    elif args.optim == "l":
        args.optim = "localization"

    if args.mode == "e":
        args.mode = "experimental"
    elif args.mode == "s":
        args.mode = "simulation"

    return args


def get_collate_func(mode):
    if mode == "experimental":
        return collate_experimental
    elif mode == "simulation":
        return collate_simulation
    else:
        raise ValueError("Wrong mode selected.")


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
        "optim",
        choices=["r", "l"],
        help="Choose which optimization problem to collate.",
    )
    parser.add_argument(
        "mode", choices=["e", "s"], help="Choose which mode to collate."
    )
    parser.add_argument("serial", help="Enter the data serial to collate.")
    args = format_args(parser.parse_args())
    path = ROOT / args.optim / args.mode / args.serial
    get_collate_func(args.mode)(path)
