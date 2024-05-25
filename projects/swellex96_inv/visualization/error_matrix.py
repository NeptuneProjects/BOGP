#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from data.bo import common, helpers

N_INIT = 64
N_SOBOL = 10000
N_TRIALS = 100


def load_data(prepend: str = "") -> list[pd.DataFrame]:
    data = []
    for glob, true_vals in zip(
        [f"{prepend}sim_*/*.npz", f"{prepend}exp_*/*.npz"],
        [common.TRUE_SIM_VALUES, common.TRUE_EXP_VALUES],
    ):
        df = helpers.load_data(
            common.SWELLEX96Paths.outputs / "runs", glob, common.SEARCH_SPACE, true_vals
        )
        df = helpers.split_sobol_results(df, N_TRIALS)

        sel = (
            ((df["Strategy"] == "Sobol (100)") & (df["Trial"] == N_TRIALS))
            | ((df["Strategy"] == "Sobol (10k)") & (df["Trial"] == N_SOBOL))
            | (
                (df["Strategy"] != "Sobol")
                & (df["n_init"] == N_INIT)
                & (df["Trial"] == N_TRIALS)
            )
        )
        df = df.loc[sel]

        sel = (df["Strategy"] == "Sobol (100)") | (df["Strategy"] == "Sobol (10k)")
        df.loc[sel, "wall_time"] = df.loc[sel, "wall_time"] * 32

        data.append(df)

    return data


def get_err_matrix(df: pd.DataFrame) -> None:
    strategies = sorted(list(df["Strategy"].unique()), key=common.SORTING_RULE.__getitem__)
    df = df.drop(
        columns=[
            "seed",
            "wall_time",
            "Trial",
            "n_iter",
            "n_init",
            "obj",
            "rec_r",
            "src_z",
            "tilt",
            "h_w",
            "h_sed",
            "c_p_sed_top",
            "dc_p_sed",
            "c_p_sed_bot",
            "best_rec_r",
            "best_src_z",
            "best_tilt",
            "best_h_w",
            "best_h_sed",
            "best_c_p_sed_top",
            "best_dc_p_sed",
            "best_c_p_sed_bot",
            "best_dc_p_sed_err",
        ]
    )

    df = df.groupby("Strategy").mean()
    df = df.reindex(strategies)
    print(df)
    # print(df.to_latex(float_format="%.2f"))


def main() -> None:
    data = load_data()
    get_err_matrix(data[0])
    get_err_matrix(data[1])


if __name__ == "__main__":
    main()
