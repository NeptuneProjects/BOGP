#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
from pathlib import Path
import sys
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from tritonoa.at.models.kraken.runner import run_kraken
from tritonoa.sp.beamforming import beamformer
from tritonoa.sp.mfp import MatchedFieldProcessor
from tritonoa.sp.processing import simulate_covariance

sys.path.insert(0, str(Path(__file__).parents[1]))
from conf import common
from data import formatter

sys.path.insert(0, str(Path(__file__).parents[3]))
from optimization import utils

SIMULATE = False
TRUE_R = 4.15
TRUE_SRC_Z = 66.3
TRUE_TILT = 0.4
TRUE_VALUES = {
    "rec_r": TRUE_R,
    "src_z": TRUE_SRC_Z,
    "h_w": 217.0,
    "h_s": 23.0,
    "c_s": 1572.3,
    "dcdz_s": 0.9,
    "tilt": TRUE_TILT,
}


def main():
    time_step = 50
    num_points = 51
    base_env = utils.load_env_from_json(common.SWELLEX96Paths.environment_data)

    if SIMULATE:
        K = simulate_covariance(
            runner=run_kraken,
            parameters=base_env
            | {"rec_r": TRUE_R, "src_z": TRUE_SRC_Z, "tilt": TRUE_TILT},
            freq=common.FREQ,
        )
    else:
        K = utils.load_covariance_matrices(
            paths=utils.get_covariance_matrix_paths(
                freq=common.FREQ, path=common.SWELLEX96Paths.acoustic_path
            ),
            index=time_step,
        )

    processor = MatchedFieldProcessor(
        runner=run_kraken,
        covariance_matrix=K,
        freq=common.FREQ,
        parameters=base_env,
        parameter_formatter=formatter.format_parameters,
        beamformer=partial(beamformer, atype="cbf_ml"),
        multifreq_method="product",
    )
    search_space = [
        {"name": "rec_r", "type": "range", "bounds": [TRUE_R - 0.5, TRUE_R + 0.5]},
        {"name": "src_z", "type": "range", "bounds": [30.0, 90.0]},
        {"name": "h_w", "type": "range", "bounds": [200.0, 240.0]},
        {"name": "h_s", "type": "range", "bounds": [1.0, 30.0]},
        {"name": "c_s", "type": "range", "bounds": [1500.0, 1800.0]},
        {"name": "dcdz_s", "type": "range", "bounds": [0.0, 4.0]},
        {"name": "tilt", "type": "range", "bounds": [-4.0, 4.0]},
    ]

    sensitivities = list()
    for i, parameter in enumerate(search_space):
        print(f"Computing {parameter['name']}")
        space = np.linspace(
            parameter["bounds"][0], parameter["bounds"][1], num_points
        ).tolist()

        if parameter["name"] == "rec_r":
            sensitivities.append(
                {
                    "name": parameter["name"],
                    "space": space,
                    "value": processor({**TRUE_VALUES} | {"rec_r": space}),
                }
            )
        else:
            B = np.zeros(num_points)
            for i, value in enumerate(
                tqdm(
                    space,
                    bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
                )
            ):
                B[i] = processor({**TRUE_VALUES} | {parameter["name"]: value})

            sensitivities.append(
                {
                    "name": parameter["name"],
                    "space": space,
                    "value": B,
                }
            )
        print(f"Complete.")

    np.save(common.SWELLEX96Paths.outputs / "sensitivity.npy", sensitivities)

    fig, axs = plt.subplots(nrows=len(sensitivities), ncols=1, figsize=(6, 8))
    for i, parameter in enumerate(sensitivities):
        if len(sensitivities) > 1:
            ax = axs[i]
        else:
            ax = plt.gca()
        B = parameter["value"]
        # B = 1 - B
        # B = 10 * np.log10(B)

        ax.plot(parameter["space"], B, label=parameter["name"])
        ax.axvline(TRUE_VALUES[parameter["name"]], color="k", linestyle="--")
        # ax.set_ylim([-10, 0.1])
        ax.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
