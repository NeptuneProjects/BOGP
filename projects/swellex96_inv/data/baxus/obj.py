# -*- coding: utf-8 -*-

from functools import partial
from pathlib import Path
import sys

from tritonoa.at.models.kraken.runner import run_kraken
from tritonoa.sp.beamforming import beamformer
from tritonoa.sp.mfp import MatchedFieldProcessor

sys.path.insert(0, str(Path(__file__).parents[2]))
from conf import common
from data import formatter

sys.path.insert(0, str(Path(__file__).parents[4]))
from optimization import utils


def get_objective() -> MatchedFieldProcessor:
    base_env = utils.load_env_from_json(common.SWELLEX96Paths.simple_environment_data)
    return MatchedFieldProcessor(
        runner=run_kraken,
        covariance_matrix=utils.load_covariance_matrices(
            paths=utils.get_covariance_matrix_paths(
                freq=common.FREQ, path=common.SWELLEX96Paths.acoustic_path
            ),
            index=common.TIME_STEP,
        ),
        freq=common.FREQ,
        parameters=base_env,
        parameter_formatter=formatter.format_parameters,
        beamformer=partial(beamformer, atype="cbf_ml"),
        multifreq_method="product",
    )
