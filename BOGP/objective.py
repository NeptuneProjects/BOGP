#!/usr/bin/env python3

import numpy as np

from tritonoa.kraken import run_kraken
from tritonoa.sp import beamformer


class MatchedFieldProcessor:
    """For use in BoTorch training example."""
    def __init__(self, K, parameters, atype="cbf"):
        self.K = K
        self.parameters = parameters
        self.atype = atype

    def __call__(self, parameters):
        return self.evaluate_true(parameters)

    def __str__(self):
        return self.__class__.__name__

    def evaluate_true(self, parameters):
        p_rep = run_kraken(self.parameters | parameters)
        return abs(beamformer(self.K, p_rep, atype=self.atype).item())

    def _get_name(self):
        return self.__class__.__name__


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
