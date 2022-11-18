#!/usr/bin/env python3

from pathlib import Path
import re

import pandas as pd
from tqdm import tqdm

from .optimization.optimizer import Results


class Aggregator:
    def __init__(self, path):
        if isinstance(path, str):
            self.path = Path(path)
        else:
            self.path = path
        
        self.pbar_kwargs = {
            "desc": "Loading results",
            "bar_format": "{l_bar}{bar:20}{r_bar}{bar:-20b}",
            "leave": True,
            "position": 0,
            "unit": "files",
            "colour": "blue"
        }


class BayesianOptimizationGPAggregator(Aggregator):
    def __init__(self, path):
        super().__init__(path)
        self.data = {
            "acq_func": [],
            "snr": [],
            "rec_r": [],
            "seed": [],
            "evaluation": [],
            "best_value": [],
            "best_param": [],
        }

    def run(self, savepath=None, verbose=False):
        if savepath is None:
            savepath = Path.cwd()
        df = self.read(verbose=verbose)
        df.to_csv(savepath)

    def read(self, verbose=False):
        files = sorted([f for f in (self.path / "Runs").glob("*/results.pth")])
        folder_pairs = re.split("=|__", str(self.path))
        folder_values = [folder_pairs[i + 1] for i in range(0, len(folder_pairs), 2)]

        for f in tqdm(files, disable=not verbose, **self.pbar_kwargs):
            results = Results().load(f)
            for i, result in enumerate(results):
                self.data["acq_func"].append(folder_values[0])
                self.data["snr"].append(folder_values[1])
                self.data["rec_r"].append(folder_values[2])
                self.data["seed"].append(f.parent.stem)
                self.data["evaluation"].append(i)
                self.data["best_value"].append(result.best_value)
                # TODO: Need to parse multiple parameters
                self.data["best_param"].append(result.best_parameters.numpy())

        return pd.DataFrame(self.data)


class RandomSearchAggregator(Aggregator):
    def __init__(self, path):
        super().__init__(path)
        self.data = None
        # TODO: Write aggregator class for random search