# -*- coding: utf-8 -*-

import logging
from pathlib import Path
from typing import Generator, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.quasirandom import SobolEngine


def get_random_seeds(n: int) -> list[int]:
    return np.random.choice(np.arange(0, 9999), size=n, replace=False).tolist()


def get_initial_points(
    dim: int, n_pts: int, dtype: torch.dtype, device: torch.device, seed=0
) -> torch.Tensor:
    sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
    # points have to be in [-1, 1]^d
    return 2 * sobol.draw(n=n_pts).to(dtype=dtype, device=device) - 1


def initialize_logger_file(
    fname: Path, logger: logging.Logger, logfmt: logging.Formatter
) -> None:
    fh = logging.FileHandler(filename=fname)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logfmt)
    logger.addHandler(fh)


def initialize_std_logger(logger: logging.Logger, logfmt: logging.Formatter) -> None:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(logfmt)
    logger.addHandler(ch)


def get_bounds_from_search_space(search_space: list[dict]) -> np.ndarray:
    return np.array([(d["bounds"][0], d["bounds"][1]) for d in search_space])


def transform_to_original_space(X: np.ndarray, search_space: list[dict]) -> np.ndarray:
    bounds = get_bounds_from_search_space(search_space)
    return bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * (X + 1) / 2


def get_best_params(X: np.ndarray, Y: np.ndarray, search_space: dict) -> np.ndarray:
    best_params = X[np.argmin(Y, axis=0)]
    return transform_to_original_space(best_params, search_space)


def log_current_value_and_parameters(
    X: np.ndarray, Y: np.ndarray, search_space: dict, strategy: Optional[str] = None
) -> None:
    logging.info(
        f"[Current] Value: {Y[-1].item():.5f} | Parameters: "
        + "".join(
            [
                f"{d['name']}: {v:.2f} | "
                for d, v in zip(
                    search_space,
                    transform_to_original_space(X[-1], search_space).squeeze(),
                )
            ]
        )[:-3]
    )


def log_best_value_and_parameters(
    X: np.ndarray, Y: np.ndarray, search_space: dict
) -> None:
    best_params = get_best_params(X, Y, search_space)
    logging.info(
        f"   [Best] Value: {Y.min().item():.5f} | Parameters: "
        + "".join(
            [
                f"{d['name']}: {v:.2f} | "
                for d, v in zip(search_space, best_params.squeeze())
            ]
        )[:-3]
    )


def parse_name(name: str) -> tuple[str, int, int, str]:
    strategy_names = {
        "ei": "EI",
        "pi": "PI",
        "ucb": "UCB",
        "sobol": "Sobol",
        "baxus": "BAxUS",
        "random": "Random",
    }

    parts = name.strip(".npz").split("_")
    strategy = parts[0]
    parts1 = parts[1].split("-")
    n_iter = int(parts1[0])
    n_init = int(parts1[1])
    seed = parts[2]

    return strategy_names[strategy], n_iter, n_init, seed


def record_best_evaluations(
    df: pd.DataFrame, search_space: list[dict], true_values: dict
) -> pd.DataFrame:
    search_parameters = [d["name"] for d in search_space] + [
        "c_p_sed_bot",
        # "c2",
        # "c3",
        "c4",
        # "c5",
        # "c6",
    ]
    # Iterate through the dataframe and record the running best evaluation and the corresponding parameters.
    best_obj = np.inf
    best_params = np.zeros(len(search_parameters))
    best_obj_history = []
    best_params_history = []
    for i in range(df.shape[0]):
        if df["obj"][i] < best_obj:
            best_obj = df["obj"][i]
            best_params = df[search_parameters].iloc[i].values
        best_obj_history.append(best_obj)
        best_params_history.append(best_params)

    df["best_obj"] = best_obj_history
    for i, param in enumerate(search_parameters):
        df["best_" + param] = np.array(best_params_history)[:, i]
        df["best_" + param + "_err"] = np.abs(df[param] - true_values[param])

    return df


def construct_run_df(
    f: Path, search_space: list[dict], true_values: dict
) -> pd.DataFrame:
    search_parameters = [d["name"] for d in search_space]
    columns = (
        [
            "Strategy",
            "Trial",
            "n_iter",
            "n_init",
            "seed",
            "wall_time",
            "obj",
            "best_obj",
        ]
        + search_parameters
        + [("best_" + p) for p in search_parameters]
        + [("best_" + p + "_err") for p in search_parameters]
    )
    data = np.load(f)
    X = transform_to_original_space(data["X"], search_space=search_space)
    Y = data["Y"]
    t = data["t"]

    df = pd.DataFrame(columns=columns)
    strategy, n_iter, n_init, seed = parse_name(f.name)

    df["Trial"] = np.arange(X.shape[0]) + 1

    for param, j in zip(search_parameters, range(X.shape[1])):
        df[param] = X[:, j]

    df = compute_c_p_sed_bot(df)
    df = compute_ssp(df)

    df["obj"] = Y
    df["Strategy"] = strategy
    df["n_iter"] = n_iter
    df["n_init"] = n_init
    df["seed"] = seed
    # df["wall_time"] = pd.TimedeltaIndex(np.cumsum(t), unit="s")
    df["wall_time"] = np.cumsum(t)
    df = record_best_evaluations(df, search_space, true_values)

    return df


def compute_c_p_sed_bot(df: pd.DataFrame) -> pd.DataFrame:
    df["c_p_sed_bot"] = df["c_p_sed_top"] + df["dc_p_sed"]
    df["best_c_p_sed_bot"] = np.nan
    df["best_c_p_sed_bot_err"] = np.nan
    return df


def compute_ssp(df: pd.DataFrame) -> pd.DataFrame:

    df["c4"] = df["c3"]
    df["best_c4"] = np.nan
    df["best_c4_err"] = np.nan

#     df["c2"] = 1522.0 + df["dc1"]
#     df["best_c2"] = np.nan
#     df["best_c2_err"] = np.nan

#     df["c3"] = df["c2"] + df["dc2"]
#     df["best_c3"] = np.nan
#     df["best_c3_err"] = np.nan

#     df["c4"] = df["c3"] + df["dc3"]
#     df["best_c4"] = np.nan
#     df["best_c4_err"] = np.nan

#     df["c5"] = df["c4"] + df["dc4"]
#     df["best_c5"] = np.nan
#     df["best_c5_err"] = np.nan

#     df["c6"] = df["c5"] + df["dc5"]
#     df["best_c6"] = np.nan
#     df["best_c6_err"] = np.nan
    return df


def construct_agg_df(
    files: Generator, search_space: list[dict], true_values: dict
) -> pd.DataFrame:
    return pd.concat([construct_run_df(f, search_space, true_values) for f in files])


def load_data(
    path: Path, pattern: str, search_space: list[dict], true_values: dict
) -> pd.DataFrame:
    return construct_agg_df(path.glob(pattern), search_space, true_values)


def adjust_subplotxticklabels(
    ax: plt.Axes, low: Optional[int] = None, high: Optional[int] = None
) -> None:
    ticklabels = ax.get_xticklabels()
    if low is not None:
        ticklabels[low].set_ha("left")
    if high is not None:
        ticklabels[high].set_ha("right")


def adjust_subplotyticklabels(
    ax: plt.Axes, low: Optional[int] = None, high: Optional[int] = None
) -> None:
    ticklabels = ax.get_yticklabels()
    if low is not None:
        ticklabels[low].set_va("bottom")
    if high is not None:
        ticklabels[high].set_va("top")
