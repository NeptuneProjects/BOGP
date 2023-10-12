#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial
from pathlib import Path
import sys

from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from tritonoa.at.models.kraken.runner import run_kraken
from tritonoa.sp.beamforming import beamformer
from tritonoa.sp.mfp import MatchedFieldProcessor

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
    # {"name": "c1", "type": "range", "bounds": [1470.0, 1570.0]},
    {"name": "dc1", "type": "range", "bounds": [-40.0, 20.0]},
    {"name": "dc2", "type": "range", "bounds": [-10.0, 10.0]},
    {"name": "dc3", "type": "range", "bounds": [-10.0, 10.0]},
    {"name": "dc4", "type": "range", "bounds": [-5.0, 5.0]},
    {"name": "dc5", "type": "range", "bounds": [-5.0, 5.0]},
    {"name": "dc6", "type": "range", "bounds": [-5.0, 5.0]},
    # {"name": "dc7", "type": "range", "bounds": [-5.0, 5.0]},
    # {"name": "dc8", "type": "range", "bounds": [-5.0, 5.0]},
    # {"name": "dc9", "type": "range", "bounds": [-5.0, 5.0]},
    # {"name": "dc10", "type": "range", "bounds": [-10.0, 10.0]},
    # {"name": "h_w", "type": "range", "bounds": [200.0, 240.0]},
    # {"name": "h_s", "type": "range", "bounds": [1.0, 100.0]},
    # {"name": "bot_c_p", "type": "range", "bounds": [1500.0, 1700.0]},
    # {"name": "bot_rho", "type": "range", "bounds": [1.0, 3.0]},
    # {"name": "dcdz_s", "type": "range", "bounds": [0.0, 3.0]},
    {"name": "tilt", "type": "range", "bounds": [-4.0, 4.0]},
]


def objective(x):
    parameters = {d["name"]: x[i] for i, d in enumerate(search_space)}
    return processor(parameters).item()


kernel = NUTS(potential_fn=processor)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=3)
