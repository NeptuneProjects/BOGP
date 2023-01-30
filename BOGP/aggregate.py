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
        self.savename = self.savepath / "aggregated_results.csv"
        self.files = list(self.path.glob("*/*/*/results.json"))

    def run(self, verbose=True):
        os.makedirs(self.savepath, exist_ok=True)
        for fname in tqdm(
            self.files, total=len(self.files), disable=not verbose, **self.pbar_kwargs
        ):
            df = self.extract_results(fname)
            df.to_csv(
                self.savename,
                mode="a",
                header=not self.savename.exists(),
            )

    @staticmethod
    def extract_results(fname):
        client = AxClient.load_from_json_file(fname, verbose_logging=False)
        values_to_append = client.experiment.name.split(";")
        df = client.get_trials_data_frame()
        for k, v in zip(COLUMNS, values_to_append):
            df[k] = v
        cols = df.columns.to_list()
        cols = cols[-6:] + cols[:-6]
        return df[cols]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()
    agg = Aggregator(args.path)
    agg.run()
