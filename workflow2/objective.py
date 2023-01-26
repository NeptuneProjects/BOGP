#!/usr/bin/env python3

from tritonoa.kraken import run_kraken
from tritonoa.sp import beamformer

def objective_function(parameters):
    K = parameters.pop("K")
    env_parameters = parameters.pop("env_parameters")
    p_rep = run_kraken(parameters | env_parameters)
    
    objective = beamformer(K, p_rep, atype="cbf").item()
    return {"bartlett": (objective, 0.0)}
