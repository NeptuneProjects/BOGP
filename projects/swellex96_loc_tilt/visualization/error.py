#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from statistics import mean, median
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[1]))
from conf.common import SWELLEX96Paths

SOURCE_CSV = (
    SWELLEX96Paths.outputs
    / "loc_tilt"
    / "experimental"
    / "serial_009"
    / "results"
    / "collated_results.csv"
)
SKIP_REGIONS = [
    list(range(37, 43)),
    list(range(87, 91)),
    list(range(95, 98)),
]
skip_steps = sum(SKIP_REGIONS, [])


def compute_strategy_mae(df_tow: pd.DataFrame, df_res: pd.DataFrame) -> pd.DataFrame:
    df_tow = df_tow[~df_tow.index.isin(skip_steps)]
    df_res = df_res[~df_res["Time Step"].isin(skip_steps)]

    rows = []
    for strategy in df_res.strategy.unique():
        selection = df_res["strategy"] == strategy
        df_sel = df_res[selection].copy()
        range_mae = mean(
            abs(df_sel["best_rec_r"].values - df_tow["Apparent Range [km]"].values)
        )
        depth_mae = mean(
            abs(df_sel["best_src_z"].values - df_tow["Apparent Depth [m]"].values)
        )
        tilt_mae = mean(
            abs(df_sel["best_tilt"].values - df_tow["Apparent Tilt [deg]"].values)
        )
        rows.append(
            {
                "Strategy": strategy,
                "$r_\mathrm{src}$ [km]": range_mae,
                "$z_\mathrm{src}$ [m]": depth_mae,
                "$\tau$ [deg]": tilt_mae,
            }
        )

    return pd.DataFrame(data=rows).round(3)


def compute_tilt_mae(
    df_tow: pd.DataFrame, data: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    df_tow = df_tow[~df_tow.index.isin(skip_steps)]

    rows = []
    for k, df in data.items():
        df = df[~df["Time Step"].isin(skip_steps)]
        range_mae = mean(
            abs(df["best_rec_r"].values - df_tow["Apparent Range [km]"].values)
        )
        depth_mae = mean(
            abs(df["best_src_z"].values - df_tow["Apparent Depth [m]"].values)
        )
        mean_corr = df["best_value"].mean()
        rows.append(
            {
                "$\tau$": k,
                "$r_\mathrm{src}$ MAE [km]": range_mae,
                "$z_\mathrm{src}$ MAE [m]": depth_mae,
                "Mean $\hat{\phi}$": mean_corr,
            }
        )

    return pd.DataFrame(data=rows).round(3)


def main() -> None:
    df_tow = pd.read_csv(SWELLEX96Paths.gps_data)
    df_res = pd.read_csv(SOURCE_CSV)
    error = compute_strategy_mae(df_tow, df_res)
    print(error.to_latex(index=False, escape=False))

    # df_var = df_res[(df_res["strategy"] == "Sobol+GP/EI")]
    # df_0 = pd.read_csv(SOURCE_CSV.parents[3] / "serial_012" / "results" / "collated_results.csv")
    # df_1 = pd.read_csv(SOURCE_CSV.parents[3] / "serial_013" / "results" / "collated_results.csv")
    # df_2 = pd.read_csv(SOURCE_CSV.parents[3] / "serial_014" / "results" / "collated_results.csv")
    # data = {
    #     "Est.": df_var,
    #     "$0^\circ$": df_0,
    #     "$1^\circ$": df_1,
    #     "$2^\circ$": df_2
    # }
    # compute_tilt_mae(df_tow, data)


if __name__ == "__main__":
    main()
