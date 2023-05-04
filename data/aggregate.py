#!/usr/bin/env python3

from argparse import ArgumentParser
import logging
from pathlib import Path
from typing import Union

import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Aggregator:
    # TODO: Add parameterization to dataframes.
    pbar_kwargs = {
        "desc": "Aggregating results",
        "bar_format": "{l_bar}{bar:20}{r_bar}{bar:-20b}",
        "leave": True,
        "position": 0,
        "unit": "files",
        "colour": "blue",
    }

    def __init__(self, serial_path: Union[Path, str]) -> None:
        self.serial_path = Path(serial_path)
        self.savepath = self.serial_path / "results"
        self.savename_agg = self.savepath / "aggregated_results.csv"
        self.savename_best = self.savepath / "best_results.csv"
        self.files = list(self.serial_path.glob("*/*/*/results.csv"))

    @staticmethod
    def append_strategy_seed_cols(df, strategy, seed):
        df["strategy"] = strategy
        df["seed"] = seed
        return df

    def build_best_df(self) -> pd.DataFrame:
        def _format_df(path: Path) -> pd.DataFrame:
            return self.append_strategy_seed_cols(
                pd.read_csv(path, index_col=0).tail(1), *path.parts[-3:-1]
            )

        return self.remove_parameter_columns(
            self.remove_duplicate_columns(
                pd.concat(map(_format_df, [f for f in self.files]), ignore_index=True)
            )
        )

    def build_full_df(self) -> pd.DataFrame:
        def _format_df(path: Path) -> pd.DataFrame:
            return self.append_strategy_seed_cols(
                pd.read_csv(path, index_col=0), *path.parts[-3:-1]
            )

        return self.remove_duplicate_columns(
            pd.concat(map(_format_df, [f for f in self.files]), ignore_index=True)
        )

    def mkdir(self) -> None:
        self.savepath.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=[col for col in df.columns if col.endswith(".1")])

    @staticmethod
    def remove_parameter_columns(df: pd.DataFrame) -> pd.DataFrame:
        best_columns = [col for col in df.columns if col.startswith("best_")]
        columns_to_remove = [col for col in df.columns if f"best_{col}" in best_columns]
        return df.drop(columns=[col for col in df.columns if col.endswith(".1")])

    def run(self) -> None:
        logger.warning(f"Aggregating results from {self.serial_path}")
        self.mkdir()
        logger.warning(f"Created destination folder: {self.savepath}")
        df_full = self.build_full_df()
        logger.warning("Built full results dataframe.")
        df_full.to_csv(self.savename_agg)
        logger.warning(f"Saved full results dataframe to {self.savename_agg}")
        df_best = self.build_best_df()
        logger.warning("Built best results dataframe.")
        df_best.to_csv(self.savename_best)
        logger.warning(f"Saved best results dataframe to {self.savename_best}")

    # COLUMNS = ["optimization", "mode", "serial", "scenario", "strategy", "seed"]
    # @classmethod
    # def extract_results(cls, fname):
    #     client = AxClient.load_from_json_file(fname, verbose_logging=False)
    #     values_to_append = client.experiment.name.split(";")
    #     df = client.get_trials_data_frame()
    #     df = cls.get_best_results(df, client)
    #     for k, v in zip(COLUMNS, values_to_append):
    #         df[k] = v
    #     cols = df.columns.to_list()
    #     cols = cols[-6:] + cols[:-6]
    #     df = df[cols]

    #     best_index = df.iloc[-1]["best_trial"]
    #     df_best = df.iloc[[best_index]]
    #     df_best = df_best.drop(columns=["best_trial", "best_parameters", "best_values"])

    #     return df, df_best

    # @staticmethod
    # def get_best_results(df, client):
    #     best_value = 0
    #     best_i = 0
    #     best_trial = []
    #     best_values = []
    #     best_params = []
    #     for i, row in df.iterrows():
    #         value = row[client.objective_name]

    #         if value > best_value:
    #             best_i = i
    #             best_value = value
    #             best_param = client.get_trial_parameters(i)

    #         best_trial.append(best_i)
    #         best_params.append(best_param)
    #         best_values.append(best_value)

    #     df["best_trial"] = best_i
    #     df["best_parameters"] = best_params
    #     df["best_values"] = best_values
    #     return df


def main(args):
    agg = Aggregator(args.path)
    "../data/swellex96_S5_VLA/outputs/localization/simulation/serial_000"
    agg.run()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()
    main(args)
