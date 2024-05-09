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

NUM_POINTS = 51
PLOT = True


def compute_sensitivity(simulate=True, num_points=51):
    if simulate:
        environment = utils.load_env_from_json(
            common.SWELLEX96Paths.main_environment_data_sim
        )
        fixed_parameters = common.TRUE_SIM_VALUES
        K = simulate_covariance(
            runner=run_kraken,
            parameters=environment
            | {
                "rec_r": fixed_parameters["rec_r"],
                "src_z": fixed_parameters["src_z"],
                "tilt": fixed_parameters["tilt"],
            },
            freq=common.FREQ,
        )
    else:
        environment = utils.load_env_from_json(
            common.SWELLEX96Paths.main_environment_data_exp
        )
        fixed_parameters = common.TRUE_EXP_VALUES
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
        parameters=environment,
        parameter_formatter=param_map.format_parameters,
        beamformer=partial(beamformer, atype="cbf_ml"),
        multifreq_method="product",
    )
    search_space = common.SEARCH_SPACE

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
                    "value": processor({**fixed_parameters} | {"rec_r": space}),
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
                B[i] = processor({**fixed_parameters} | {parameter["name"]: value})

            sensitivities.append(
                {
                    "name": parameter["name"],
                    "space": space,
                    "value": B,
                }
            )
        print(f"Complete.")

    if simulate:
        savename = "sensitivity_sim.npy"
    else:
        savename = "sensitivity_exp.npy"
    np.save(common.SWELLEX96Paths.outputs / savename, sensitivities)

    return sensitivities


def plot_sensitivity(sensitivities: np.ndarray, parameters: dict):
    fig, axs = plt.subplots(nrows=len(sensitivities), ncols=1, figsize=(6, 12))
    for i, parameter in enumerate(sensitivities):
        if len(sensitivities) > 1:
            ax = axs[i]
        else:
            ax = plt.gca()
        B = parameter["value"]

        ax.plot(parameter["space"], B, label=parameter["name"])
        ax.axvline(parameters[parameter["name"]], color="k", linestyle="--")
        ax.legend()

    fig.suptitle(f"{common.FREQ} Hz")
    plt.tight_layout()
    plt.draw()


def main():
    sensitivities_sim = compute_sensitivity(simulate=True, num_points=NUM_POINTS)
    sensitivities_exp = compute_sensitivity(simulate=False, num_points=NUM_POINTS)
    if PLOT:
        plot_sensitivity(sensitivities_sim, common.TRUE_SIM_VALUES)
        plot_sensitivity(sensitivities_exp, common.TRUE_EXP_VALUES)
        plt.show()


if __name__ == "__main__":
    main()
