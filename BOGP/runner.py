#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import shutil
from typing import Union

import numpy as np

from oao.common import UNINFORMED_STRATEGIES
from oao.handler import Handler
from oao.utilities import load_config, set_device
from objective import objective_function
from tritonoa.kraken import clean_up_kraken_files


def config_destination_path(config, path):
    if config["strategy"]["loop_type"] not in UNINFORMED_STRATEGIES:
        if (
            config["strategy"]["generation_strategy"]
            ._steps[1]
            .model_kwargs["botorch_acqf_class"]
            .__name__
            == "qExpectedImprovement"
        ):
            acqf = "_qei"
        elif (
            config["strategy"]["generation_strategy"]
            ._steps[1]
            .model_kwargs["botorch_acqf_class"]
            .__name__
            == "qProbabilityOfImprovement"
        ):
            acqf = "_qpi"
        elif (
            config["strategy"]["generation_strategy"]
            ._steps[1]
            .model_kwargs["botorch_acqf_class"]
            .__name__
            == "qUpperConfidenceBound"
        ):
            acqf = "_qucb"
        loop_type = config["strategy"]["loop_type"] + acqf
    else:
        loop_type = config["strategy"]["loop_type"] + ""

    destination = path / loop_type / f"seed_{config['seed']:010d}"
    os.makedirs(destination, exist_ok=True)
    return destination


def main(config_path, device):
    if isinstance(config_path, str):
        config_path = Path(config_path)
    config = load_config(config_path)
    root_path = config_path.parent.parent
    scenario_path = root_path / Path(config["destination"])
    destination = config_destination_path(config, scenario_path)
    config["obj_func_parameters"]["env_parameters"]["tmpdir"] = destination

    if config["strategy"]["loop_type"] not in UNINFORMED_STRATEGIES:
        config["strategy"]["generation_strategy"]._steps[1].model_kwargs[
            "torch_device"
        ] = set_device(device)

    K = np.load(scenario_path / "data" / "measurement_covariance.npy")
    config["obj_func_parameters"]["K"] = K

    Handler(config, destination, objective_function).run()
    clean_up_kraken_files(destination)
    shutil.move(config_path.absolute(), (destination / config_path.name).absolute())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    main(args.path, args.device)
