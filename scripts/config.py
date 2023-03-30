#!/usr/bin/env python3

from datetime import datetime
from pathlib import Path

import tomli_w

ROOT = str((Path.cwd() / "Data").relative_to(Path.cwd()))
SEED = 2009
SERIAL = datetime.now().strftime("serial_%Y%m%dT%H%M%S")
FREQUENCIES = [148, 166, 201, 235, 283, 338, 388]

config = {
    "range_estimation": {
        "simulation": {
            "num_mc_runs": 100,
            "num_restarts": 20,
            "num_samples": 512,
            "num_trials": 100,
            "num_warmup": 5,
            "q": 5,
            "root": ROOT,
            "main_seed": SEED,
            "serial": "serial_example",
            "evaluation_config": {"num_test_points": 100},
            "frequencies": [201]
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
            "serial": SERIAL,
            "frequencies": FREQUENCIES
        },
    },
    "localization": {
        "simulation": {
            "num_mc_runs": 100,
            "num_restarts": 40,
            "num_samples": 1024,
            "num_trials": 800,
            "num_grid_trials": [40, 20],
            "num_warmup": 50,
            "q": 5,
            "root": ROOT,
            "main_seed": SEED,
            "serial": SERIAL,
            "frequencies": FREQUENCIES
        },
        "experimental": {
            "num_mc_runs": 1,
            "num_restarts": 40,
            "num_samples": 1024,
            "num_trials": 800,
            "num_grid_trials": [40, 20],
            "num_warmup": 50,
            "q": 5,
            "root": ROOT,
            "main_seed": SEED,
            "serial": SERIAL,
            "frequencies": FREQUENCIES
        }
    }
}

if __name__ == "__main__":
    config_path = (Path.cwd() / "Source" / "scripts" / "config.toml").relative_to(Path.cwd())
    with open(config_path, "wb") as fp:
        tomli_w.dump(config, fp)
