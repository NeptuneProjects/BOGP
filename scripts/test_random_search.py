#!/usr/bin/env python3
#%%
from pathlib import Path
import random
import sys

from botorch.test_functions import Rastrigin
import torch

sys.path.insert(0, str(Path.cwd()))
from BOGP.optimization.optimizer import RandomSearchConfig, RandomSearch
from BOGP.acoustics import MatchedFieldProcessor


search_parameters = [
    {"name": "x1", "bounds": [-5.12, 5.12]},
    {"name": "x2", "bounds": [-5.12, 5.12]},
    {"name": "x3", "bounds": [-5.12, 5.12]},
]
obj_func_kwargs = {"dim": len(search_parameters), "negate": True}
obj_func = Rastrigin(**obj_func_kwargs)

random_search_config = RandomSearchConfig()
random_searcher = RandomSearch(
    random_search_config,
    obj_func=obj_func,
    search_parameters=search_parameters
)

X, y = random_searcher.run()
print(X)
print(y)

# bounds = random_searcher.get_bounds(search_parameters)

# def get_candidates(M, bounds):
#     """
#     (r2 - r1) * torch.rand(M, N) + r1
#     """
#     N = bounds.shape[1]
#     interval = (bounds[1] - bounds[0]).unsqueeze(-1)
#     print(interval.shape)
#     X_new = interval * torch.rand(N, M) + bounds[0].unsqueeze(-1)
#     return X_new


# X_new = get_candidates(10, bounds).T
# parameters = {
#     item["name"]: X_new[..., i] for i, item in enumerate(search_parameters)
# }
# print(parameters)


#%%
