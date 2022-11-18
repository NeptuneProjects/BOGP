#!/usr/bin/env python3

from dataclasses import dataclass, field

from botorch.test_functions import Hartmann


@dataclass
class RandomSearchConfig:
    main_seed: int = field(default=2009)
    n_total: int = field(default=10)


class RandomSearch:
    def __init__(
            self,
            config: RandomSearchConfig,
            obj_func: base.BaseTestProblem,
            search_parameters: dict,
            fixed_parameters: dict = {},
            obj_func_kwargs: dict = {},
    ):
        self.config = config
        self.obj_func: ObjectiveFunction(obj_func)
        try:
            self.obj_func_name = obj_func._get_name()
        except AttributeError:
            self.obj_func_name = obj_func.__name__
        self.obj_func_module = obj_func.__module__
        self.search_parameters = search_parameters
        self.fixed_parameters = fixed_parameters
        self.obj_func_kwargs = obj_func_kwargs

