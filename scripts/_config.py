#!/usr/bin/env python3

from datetime import datetime
from pathlib import Path

import tomli_w

ROOT = str((Path.cwd() / "Data").relative_to(Path.cwd()))
SEED = 2009
SERIAL = datetime.now().strftime("serial_%Y%m%dT%H%M%S")

config = {
    "range_estimation": {
        "simulation": {
            "num_mc_runs": 100,
            "num_restarts": 20,
            "num_samples": 512,
            "num_trials": 200,
            "num_warmup": 10,
            "q": 5,
            "root": ROOT,
            "main_seed": SEED,
            "serial": SERIAL
        },
        "experimental": {
            "num_mc_runs": 1,
            "num_restarts": 20,
            "num_samples": 512,
            "num_trials": 200,
            "num_warmup": 10,
            "q": 5,
            "root": ROOT,
            "main_seed": SEED,
            "serial": SERIAL
        },
    },
    "localization": {
        "simulation": {
            "num_mc_runs": 100,
            "num_restarts": 40,
            "num_samples": 1024,
            "num_trials": 800,
            "num_grid_trials": [20, 40],
            "num_warmup": 50,
            "q": 5,
            "root": ROOT,
            "main_seed": SEED,
            "serial": SERIAL
        },
        "experimental": {
            "num_mc_runs": 1,
            "num_restarts": 40,
            "num_samples": 1024,
            "num_trials": 800,
            "num_grid_trials": [20, 40],
            "num_warmup": 50,
            "q": 5,
            "root": ROOT,
            "main_seed": SEED,
            "serial": SERIAL
        }
    }
}

if __name__ == "__main__":
    config_path = (Path.cwd() / "Source" / "scripts" / "config.toml").relative_to(Path.cwd())
    with open(config_path, "wb") as fp:
        tomli_w.dump(config, fp)
