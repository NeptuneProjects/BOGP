#!/usr/bin/env python3

from datetime import datetime
from pathlib import Path
import tomli

import tomli_w

ROOT = str((Path.cwd() / "Data").relative_to(Path.cwd()))
SEED = 2009
SERIAL = datetime.now().strftime("serial_%Y%m%dT%H%M%S")

config = {
    "range_estimation": {
        "simulation": {
            "NUM_MC_RUNS": 100,
            "NUM_RESTARTS": 20,
            "NUM_SAMPLES": 512,
            "NUM_TRIALS": 200,
            "NUM_WARMUP": 10,
            "Q": 5,
            "ROOT": ROOT,
            "SEED": SEED,
            "SERIAL": SERIAL
        },
        "experimental": {
            "NUM_MC_RUNS": 100,
            "NUM_RESTARTS": 20,
            "NUM_SAMPLES": 512,
            "NUM_TRIALS": 200,
            "NUM_WARMUP": 10,
            "Q": 5,
            "ROOT": ROOT,
            "SEED": SEED,
            "SERIAL": SERIAL
        },
    },
    "localization": {
        "simulation": {
            "NUM_MC_RUNS": 100,
            "NUM_RESTARTS": 40,
            "NUM_SAMPLES": 1024,
            "NUM_TRIALS": 800,
            "NUM_GRID_TRIALS": [20, 40],
            "NUM_WARMUP": 20,
            "Q": 5,
            "ROOT": ROOT,
            "SEED": SEED,
            "SERIAL": SERIAL
        },
        "experimental": {
            "NUM_MC_RUNS": 100,
            "NUM_RESTARTS": 40,
            "NUM_SAMPLES": 1024,
            "NUM_TRIALS": 800,
            "NUM_GRID_TRIALS": [20, 40],
            "NUM_WARMUP": 20,
            "Q": 5,
            "ROOT": ROOT,
            "SEED": SEED,
            "SERIAL": SERIAL
        }
    }
}

# config = {
#     "range_estimation": {
#         "NUM_MC_RUNS": 100,
#         "NUM_RESTARTS": 20,
#         "NUM_SAMPLES": 512,
#         "NUM_TRIALS": 200,
#         "NUM_WARMUP": 10,
#         "Q": 5,
#         "ROOT": ROOT,
#         "SEED": SEED,
#         "SERIAL": SERIAL
#     },
#     "localization": {
#         "NUM_MC_RUNS": 100,
#         "NUM_RESTARTS": 40,
#         "NUM_SAMPLES": 1024,
#         "NUM_TRIALS": 800,
#         "NUM_GRID_TRIALS": [20, 40],
#         "NUM_WARMUP": 20,
#         "Q": 5,
#         "ROOT": ROOT,
#         "SEED": SEED,
#         "SERIAL": SERIAL
#     }
# }

if __name__ == "__main__":
    config_path = (Path.cwd() / "Source" / "scripts" / "config.toml").relative_to(Path.cwd())
    with open(config_path, "wb") as fp:
        tomli_w.dump(config, fp)
    
    with open(config_path, "rb") as fp:
        config_ = tomli.load(fp)
    
        print(config_)
