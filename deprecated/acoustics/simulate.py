#!/usr/bin/env python3

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, "/Users/williamjenkins/Research/Code/TritonOA/")

from tritonoa.kraken import KRAKENParameterization


def save_sim_results(path, data):
    np.savez(path, **data)
    

def simulate(parameters=None, path=None):
    parameterization = KRAKENParameterization(parameters=parameters, path=path)
    p_rec = parameterization.run()
    K = p_rec @ p_rec.conj().T
    save_sim_results(Path(parameterization.tmpdir / "data.npz"), {"p_rec": p_rec, "K": K})
    parameterization.write_config(
        path=(cwd / "Data" / "Experiments" / "Localization2D" / "Simulated" / "Scene001" / "ReceivedData").relative_to(cwd),
        fname="parameters"
    )


if __name__ == "__main__":
    # vv These should be handled externally - stand in for now. vv
    cwd = Path.cwd()
    path1 = (cwd / "Data" / "Experiments" / "Localization2D" / "Simulated" / "Scene001" / "ReceivedData" / "config.json").relative_to(cwd)
    path2 = (cwd / "Data" / "Experiments" / "Localization2D" / "BaseEnvironments" / "SWELLEX96" / "config.json").relative_to(cwd)
    # ^^ Stand-in only ^^
    simulate(path=[path1, path2])
