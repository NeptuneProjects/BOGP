#!/usr/bin/env python3
#%%
from pathlib import Path
import random
import sys

from botorch.test_functions import Rastrigin
import torch

sys.path.insert(0, str(Path.cwd()))
from BOGP.optimization.optimizer import RandomSearchConfig, RandomSearch, RandomResults, RandomResult
from BOGP.acoustics import MatchedFieldProcessor


search_parameters = [
    {"name": "x1", "bounds": [-5.12, 5.12]},
    {"name": "x2", "bounds": [-5.12, 5.12]},
    {"name": "x3", "bounds": [-5.12, 5.12]},
]
obj_func_kwargs = {"dim": len(search_parameters), "negate": True}
obj_func = Rastrigin(**obj_func_kwargs)

random_search_config = RandomSearchConfig(n_total=9)
random_searcher = RandomSearch(
    random_search_config,
    obj_func=obj_func,
    search_parameters=search_parameters
)

results = random_searcher.run()
print(results[-1].best_value, results[-1].best_parameters)
# result = RandomResult(X, y)
# results = RandomResults([result])
# results.save(Path.cwd() / "test.pth")


# loaded_results = RandomResults().load(Path.cwd() / "test.pth")

#%%
