#!/usr/bin/env python3

from tritonoa.kraken import run_kraken
from tritonoa.sp import beamformer

def objective_function(parameters):
    K = parameters.pop("K")
    env_parameters = parameters.pop("env_parameters")
    p_rep = run_kraken(env_parameters | parameters)
    
    objective = beamformer(K, p_rep, atype="cbf").item()
    return {"bartlett": (objective, None)}
