#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
from statistics import mean, median
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parents[2]))
from conf.swellex96.optimization.common import SWELLEX96Paths

SOURCE_CSV = (
    SWELLEX96Paths.outputs
    / "localization"
    / "experimental"
    / "serial_001"
    / "results"
    / "collated_results.csv"
)


def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def compute_mae(df: pd.DataFrame) -> pd.DataFrame:
    df_mfp = df[df["strategy"] == "High-res MFP"]
    df = df[df.strategy != "High-res MFP"]

    rows = []
    for strategy in df.strategy.unique():
        selection = df["strategy"] == strategy
        df_sel = df[selection].copy()
        range_mae = mean(abs(df_sel["best_rec_r"].values - df_mfp["best_rec_r"].values))
        range_mede = median(
            abs(df_sel["best_rec_r"].values - df_mfp["best_rec_r"].values)
        )
        depth_mae = mean(abs(df_sel["best_src_z"].values - df_mfp["best_src_z"].values))
        depth_mede = median(
            abs(df_sel["best_src_z"].values - df_mfp["best_src_z"].values)
        )
        rows.append(
            {
                "Strategy": strategy,
                "Range MAE [km]": range_mae,
                "Range Med. Error [km]": range_mede,
                "Depth MAE [m]": depth_mae,
                "Depth Med. Error [m]": depth_mede,
            }
        )

    return pd.DataFrame(data=rows).round(3)


def main() -> None:
    df = load_df(SOURCE_CSV)
    error = compute_mae(df)
    print(error.to_latex(index=False))


if __name__ == "__main__":
    main()
