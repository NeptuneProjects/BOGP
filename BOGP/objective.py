#!/usr/bin/env python3

import numpy as np

from tritonoa.kraken import run_kraken
from tritonoa.sp import beamformer


def objective_function(parameters):
    """For use in Ax production framework."""
    K = parameters.pop("K")
    frequencies = parameters.pop("frequencies")
    env_parameters = parameters.pop("env_parameters")
    B = []
    for f, k in zip(frequencies, K):
        p_rep = run_kraken(env_parameters | {"freq": f} | parameters)
        B.append(beamformer(k, p_rep, atype="cbf").item())
    objective = np.mean(np.array(B))
    return {"bartlett": (objective, 0.0)}
