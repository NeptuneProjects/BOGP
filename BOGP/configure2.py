#!/usr/bin/env python3

import os
from pathlib import Path
import swellex
from typing import Any, Protocol, Union

import numpy as np

def load_acoustic_data(datadir: Union[str, bytes, os.PathLike]) -> list[np.ndarray]:
    pass



class Initializer(Protocol):
    def configure(self) -> None:
        ...


class Experiment(Initializer):
    def __init__(self, datadir: Path, expdir: Path) -> None:
        self.datadir = datadir
        self.expdir = expdir
        self.parameters = swellex.parameters
        self.scenarios = swellex.scenarios

    def configure(self) -> None:
        raise NotImplementedError















def main():

    SIM_SCENARIOS = {"rec_r": [1.0, 3.0, 5.0, 7.0], "src_z": [60.0], "snr": [20]}
    SKIP_TIMESTEPS = (
        list(range(73, 85))
        + list(range(95, 103))
        + list(range(187, 199))
        + list(range(287, 294))
        + list(range(302, 309))
    )
    EXP_SCENARIOS = {"timestep": [i for i in range(350) if i not in SKIP_TIMESTEPS]}
    PARAMS_KWARGS = {"type": "range", "value_type": "float", "log_scale": False}
    LOCALIZATION_PARAMETERS = [
        {
            "name": "rec_r",
            "bounds": [0.01, 8.0],
            **PARAMS_KWARGS,
        },
        {
            "name": "src_z",
            "bounds": [1.0, 160.0],
            **PARAMS_KWARGS,
        },
    ]
    FREQUENCIES = swellex.HIGH_SIGNAL_TONALS[6:]
    EXP_DATADIR = (Path.cwd() / "Data" / "SWELLEX96" / "VLA" / "selected").relative_to(
        Path.cwd()
    )
    # ^^^^ Inputs
    # vvvv Outputs


    print("Hello World!")


if __name__ == "__main__":
    main()
