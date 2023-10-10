#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from functools import partial
from pathlib import Path
import sys

import numpy as np
import numpyro
from numpyro.util import enable_x64
from tritonoa.at.models.kraken.runner import run_kraken
from tritonoa.sp.beamforming import beamformer
from tritonoa.sp.mfp import MatchedFieldProcessor

from objective import hartmann6_50
from saasbo import run_saasbo

sys.path.insert(0, str(Path(__file__).parents[2]))
from conf import common
from data import formatter

sys.path.insert(0, str(Path(__file__).parents[4]))
from optimization import utils

time_step = 20
base_env = utils.load_env_from_json(common.SWELLEX96Paths.environment_data)

processor = MatchedFieldProcessor(
    runner=run_kraken,
    covariance_matrix=utils.load_covariance_matrices(
        paths=utils.get_covariance_matrix_paths(
            freq=common.FREQ, path=common.SWELLEX96Paths.acoustic_path
        ),
        index=time_step,
    ),
    freq=common.FREQ,
    parameters=base_env,
    parameter_formatter=formatter.format_parameters,
    beamformer=partial(beamformer, atype="cbf_ml"),
    multifreq_method="product",
)

if time_step == 20:
    range_space = {"name": "rec_r", "type": "range", "bounds": [5.5, 6.0]}
if time_step == 50:
    range_space = {"name": "rec_r", "type": "range", "bounds": [3.8, 4.3]}
search_space = [
    range_space,
    {"name": "src_z", "type": "range", "bounds": [40.0, 80.0]},
    {"name": "c1", "type": "range", "bounds": [1500.0, 1550.0]},
    {"name": "dc1", "type": "range", "bounds": [-40.0, 40.0]},
    {"name": "dc2", "type": "range", "bounds": [-10.0, 10.0]},
    {"name": "dc3", "type": "range", "bounds": [-5.0, 5.0]},
    {"name": "dc4", "type": "range", "bounds": [-5.0, 5.0]},
    {"name": "dc5", "type": "range", "bounds": [-5.0, 5.0]},
    {"name": "dc6", "type": "range", "bounds": [-5.0, 5.0]},
    {"name": "dc7", "type": "range", "bounds": [-5.0, 5.0]},
    {"name": "dc8", "type": "range", "bounds": [-5.0, 5.0]},
    {"name": "dc9", "type": "range", "bounds": [-5.0, 5.0]},
    # {"name": "dc10", "type": "range", "bounds": [-10.0, 10.0]},
    {"name": "h_w", "type": "range", "bounds": [200.0, 240.0]},
    # {"name": "h_s", "type": "range", "bounds": [1.0, 100.0]},
    {"name": "bot_c_p", "type": "range", "bounds": [1500.0, 1700.0]},
    # {"name": "bot_rho", "type": "range", "bounds": [1.0, 3.0]},
    # {"name": "dcdz_s", "type": "range", "bounds": [0.0, 3.0]},
    {"name": "tilt", "type": "range", "bounds": [-3.0, 3.0]},
]


def objective(x):
    parameters = {d["name"]: x[i] for i, d in enumerate(search_space)}
    out = processor(parameters).item()
    # print(type(out))
    return out


def main(args):
    lb = np.array([s["bounds"][0] for s in search_space])
    ub = np.array([s["bounds"][1] for s in search_space])
    # lb = np.zeros(50)
    # ub = np.ones(50)
    num_init_evals = 50

    X, Y = run_saasbo(
        objective,
        # hartmann6_50,
        lb,
        ub,
        args.max_evals,
        num_init_evals,
        seed=args.seed,
        alpha=0.01,
        num_warmup=512,
        num_samples=256,
        thinning=16,
        device=args.device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAASBO to estimate SSP.")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max-evals", default=200, type=int)
    parser.add_argument("--device", default="gpu", type=str, help='use "cpu" or "gpu".')
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    enable_x64()
    numpyro.set_host_device_count(1)

    main(args)
