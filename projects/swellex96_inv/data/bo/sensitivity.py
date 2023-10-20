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


sys.path.insert(0, str(Path(__file__).parents[4]))
from optimization import utils

import common, param_map

SIMULATE = False
NUM_POINTS = 51
PLOT = True


def compute_sensitivity():
    if SIMULATE:
        K = simulate_covariance(
            runner=run_kraken,
            parameters=utils.load_env_from_json(common.SWELLEX96Paths.main_environment_data)
            | {
                "rec_r": common.TRUE_VALUES["rec_r"],
                "src_z": common.TRUE_VALUES["src_z"],
                "tilt": common.TRUE_VALUES["tilt"],
            },
            freq=common.FREQ,
        )
    else:
        K = utils.load_covariance_matrices(
            paths=utils.get_covariance_matrix_paths(
                freq=common.FREQ, path=common.SWELLEX96Paths.acoustic_path
            ),
            index=common.TIME_STEP,
        )

    processor = MatchedFieldProcessor(
        runner=run_kraken,
        covariance_matrix=K,
        freq=common.FREQ,
        parameters=utils.load_env_from_json(
            common.SWELLEX96Paths.main_environment_data
        ),
        parameter_formatter=param_map.format_parameters,
        beamformer=partial(beamformer, atype="cbf_ml"),
        multifreq_method="product",
    )
    search_space = common.SEARCH_SPACE

    sensitivities = list()
    for i, parameter in enumerate(search_space):
        print(f"Computing {parameter['name']}")
        space = np.linspace(
            parameter["bounds"][0], parameter["bounds"][1], NUM_POINTS
        ).tolist()

        if parameter["name"] == "rec_r":
            sensitivities.append(
                {
                    "name": parameter["name"],
                    "space": space,
                    "value": processor({**common.TRUE_VALUES} | {"rec_r": space}),
                }
            )
        else:
            B = np.zeros(NUM_POINTS)
            for i, value in enumerate(
                tqdm(
                    space,
                    bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
                )
            ):
                B[i] = processor({**common.TRUE_VALUES} | {parameter["name"]: value})

            sensitivities.append(
                {
                    "name": parameter["name"],
                    "space": space,
                    "value": B,
                }
            )
        print(f"Complete.")

    if SIMULATE:
        savename = "sensitivity_sim.npy"
    else:
        savename = "sensitivity_exp.npy"
    np.save(common.SWELLEX96Paths.outputs / savename, sensitivities)

    return sensitivities


def plot_sensitivity(sensitivities: np.ndarray):
    fig, axs = plt.subplots(nrows=len(sensitivities), ncols=1, figsize=(6, 12))
    for i, parameter in enumerate(sensitivities):
        if len(sensitivities) > 1:
            ax = axs[i]
        else:
            ax = plt.gca()
        B = parameter["value"]

        ax.plot(parameter["space"], B, label=parameter["name"])
        ax.axvline(common.TRUE_VALUES[parameter["name"]], color="k", linestyle="--")
        ax.legend()

    fig.suptitle(f"{common.FREQ} Hz")
    plt.tight_layout()
    plt.show()
    return


def main():
    sensitivities = compute_sensitivity()
    plot_sensitivity(sensitivities)


if __name__ == "__main__":
    main()
