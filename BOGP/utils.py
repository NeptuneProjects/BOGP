#!/usr/bin/env python3

from itertools import product
from pathlib import Path

import pandas as pd


def clean_up_kraken_files(path):
    if isinstance(path, str):
        path = Path(path)

    extensions = ["env", "mod", "prt"]
    [[f.unlink() for f in path.glob(f"*.{ext}")] for ext in extensions]


def concatenate_simulation_results(paths):
    dfs = []
    for path in paths:
        try:
            dfs.append(pd.read_csv(path, index_col=0))
        except FileNotFoundError:
            pass
    return pd.concat(dfs)


def folders_of_evaluations(inp: dict) -> list:
    evals = product_of_evaluations(inp)
    return ["__".join([f"{k}={v}" for k, v in e.items()]) for e in evals]


def product_of_evaluations(inp: dict) -> list:
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))


# class Experiment:
#     def __init__(self, expparams=None, fixedparams=None, searchparams=None):
#         self.expparams = expparams
#         self.fixedparams = fixedparams
#         self.searchparams = searchparams

#     def create_folder_structure(self, root=Path.cwd()):
#         """Initialize experiment folder tree."""
#         path = []
#         for experiment in self.experiments:
#             p = Path(Path(root) / experiment)
#             p.mkdir(parents=True, exist_ok=True)
#             path.append(p)

#         return path

#     def format_exp_names(self):
#         self.experiments = [
#             "_".join(i) for i in product(*[v for v in self.expparams.values()])
#         ]
#         return self.experiments

#     def load_config(self, path):
#         config = self._read_config(path)
#         self._parse_config(config)
#         _ = self.format_exp_names()
#         return config

#     def write_config(self, path=None):
#         if path is None:
#             path = Path.cwd()
#         self._check_if_path_exists(path)
#         self._write_json(path)
#         self._write_conf(path)
    
#     @staticmethod
#     def _check_if_path_exists(path):
#         p = Path(path)
#         if not p.is_dir():
#             p.mkdir(parents=True, exist_ok=True)

#     def _parse_config(self, config):
#         self.searchparams = []
#         for i, item in enumerate(config):
#             if i == 0:
#                 self.expparams = item
#             elif i == 1:
#                 self.fixedparams = item
#             else:
#                 self.searchparams.append(item)

#     @staticmethod
#     def _read_config(path):
#         with open(path, "r") as f:
#             config = json.load(f)
#         return config

#     def _write_conf(self, path):
#         config = ConfigParser()
#         config["EXPERIMENT PARAMETERS"] = self.expparams
#         config["FIXED PARAMETERS"] = self.fixedparams
#         for i, item in enumerate(self.searchparams):
#             config[f"SEARCH PARAMETER {i}"] = item

#         with open(path / "config.ini", "w") as f:
#             config.write(f)

#     def _write_json(self, path):
#         allparams = [self.expparams, self.fixedparams]
#         for item in enumerate(self.searchparams):
#             allparams.append(item)

#         with open(path / "config.json", "w") as f:
#             json.dump(allparams, f)


# def log(path=Path.cwd(), fname="log.txt"):
#     fname = path / fname
#     if not os.path.isfile(fname):
#         open(fname, "w+").close()

#     console_logging_format = "%(levelname)s %(message)s"
#     file_logging_format = "%(levelname)s: %(asctime)s: %(message)s"
#     # Logger configuration
#     logging.basicConfig(level=logging.INFO, format=console_logging_format)
#     logger = logging.getLogger()
#     # File handler
#     handler = logging.FileHandler(fname)
#     # Set logging level for file
#     handler.setLevel(logging.INFO)
#     formatter = logging.Formatter(file_logging_format)
#     handler.setFormatter(formatter)
#     logger.addHandler(handler)
#     return logger


# def write_config(path, data):
#     write_conf(path, data)
#     write_json(path, data)

# def write_json(path, data):
#     with open(path / "config.json", "w+") as f:
#         json.dump(data, f)

# def write_conf(path, data):
#     config = ConfigParser()
#     config["PARAMETERS"] = data
#     with open(path / "config.ini", "w+") as f:
#         config.write(f)


