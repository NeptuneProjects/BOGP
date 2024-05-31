#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from data.bo import common, helpers

N_INIT = 64
N_RAND = 10000
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
        df = helpers.split_random_results(df, N_TRIALS)
        df = df.loc[df["Strategy"] != "Sobol"]

        sel = (
            ((df["Strategy"] == "Random (100)") & (df["Trial"] == N_TRIALS))
            | ((df["Strategy"] == "Random (10k)") & (df["Trial"] == N_RAND))
            | (
                ((df["Strategy"] != "Random (100)") | (df["Strategy"] != "Random (10k)"))
                & (df["n_init"] == N_INIT)
                & (df["Trial"] == N_TRIALS)
            )
        )
        df = df.loc[sel]

        sel = (df["Strategy"] == "Random (100)") | (df["Strategy"] == "Random (10k)")
        df.loc[sel, "wall_time"] = df.loc[sel, "wall_time"] * 32

        data.append(df)

    return data


def load_de_data():
    df = pd.read_csv(common.SWELLEX96Paths.outputs / "runs" / "de" / "sim_de_results.csv")
    return helpers.agg_de_data(df)


def get_err_matrix(df: pd.DataFrame) -> None:
    strategies = sorted(
        list(df["Strategy"].unique()), key=common.SORTING_RULE.__getitem__
    )
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
    print(df.to_latex(float_format="%.2f"))


def get_de_err(df: pd.DataFrame) -> None:
    for param in common.SEARCH_SPACE:
        if param["name"] == "dc_p_sed":
            pname = "c_p_sed_bot"
        else:
            pname = param["name"]
        pval = df[pname]
        df[f"{pname}_err"] = np.abs(pval - common.TRUE_SIM_VALUES[pname])
    
    df = df.drop(columns=[
        "seed",
        "nit",
        "nfev",
        "wall_time",
        "obj",
        "rec_r",
        "src_z",
        "tilt",
        "h_w",
        "h_sed",
        "c_p_sed_top",
        "dc_p_sed",
        "c_p_sed_bot",
    ]).groupby("Strategy").mean()
    print(df)
    print(df.to_latex(float_format="%.2f"))    


def main() -> None:
    # data = load_data()
    # data[0].to_csv(common.SWELLEX96Paths.outputs / "errmat_rand_sim.csv")
    # data[1].to_csv(common.SWELLEX96Paths.outputs / "errmat_rand_exp.csv")
    # get_err_matrix(data[0])
    # get_err_matrix(data[1])

    de = load_de_data()
    get_de_err(de)
    

if __name__ == "__main__":
    main()
