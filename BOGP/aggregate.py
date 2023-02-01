#!/usr/bin/env python3

from argparse import ArgumentParser
import os
from pathlib import Path

from ax.service.ax_client import AxClient
from tqdm import tqdm

import __init__

COLUMNS = ["optimization", "mode", "serial", "scenario", "strategy", "seed"]


class Aggregator:
    pbar_kwargs = {
        "desc": "Aggregating results",
        "bar_format": "{l_bar}{bar:20}{r_bar}{bar:-20b}",
        "leave": True,
        "position": 0,
        "unit": "files",
        "colour": "blue",
    }

    def __init__(self, path):
        self.path = Path(path) if isinstance(path, str) else path
        self.savepath = self.path / "results"
        self.savename_agg = self.savepath / "aggregated_results.csv"
        self.savename_best = self.savepath / "best_results.csv"
        self.files = list(self.path.glob("*/*/*/results.json"))

    def run(self, verbose=True):
        os.makedirs(self.savepath, exist_ok=True)
        for fname in tqdm(
            self.files, total=len(self.files), disable=not verbose, **self.pbar_kwargs
        ):
            df, df_best = self.extract_results(fname)
            # break
            df.to_csv(
                self.savename_agg,
                mode="a",
                header=not self.savename_agg.exists(),
            )
            df_best.to_csv(
                self.savename_best,
                mode="a",
                header=not self.savename_best.exists(),
            )
        return df, df_best

    def extract_results(self, fname):
        client = AxClient.load_from_json_file(fname, verbose_logging=False)
        values_to_append = client.experiment.name.split(";")
        df = client.get_trials_data_frame()
        df = self.get_best_results(df, client)
        for k, v in zip(COLUMNS, values_to_append):
            df[k] = v
        cols = df.columns.to_list()
        cols = cols[-6:] + cols[:-6]
        df = df[cols]

        best_index, _, _ = client.get_best_trial()
        df_best = df.loc[[best_index]]

        return df, df_best
    
    @staticmethod
    def get_best_results(df, client):
        best_value = 0
        best_values = []
        best_params = []
        for i, row in df.iterrows():
            value = row[client.objective_name]
            if value > best_value:
                best_value = value
                best_param = client.get_trial_parameters(i)
            best_params.append(best_param)
            best_values.append(best_value)
        
        df["best_parameters"] = best_params
        df["best_values"] = best_values
        return df


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()
    agg = Aggregator(args.path)
    agg.run()
