#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Protocol

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from oao.optimizer import Optimizer


class StrategyInterface(Protocol):
    def __call__(self):
        ...


class GridSearchStrategy:
    def __init__(self, num_trials=12, *args, **kwargs):
        self.num_trials = num_trials

    def __call__(self, *args, **kwargs):
        return lambda: self.num_trials


class SobolStrategy(GenerationStrategy):
    def __init__(self, num_trials=144, max_parallelism=16, seed=2009):
        super().__init__(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=num_trials,
                    max_parallelism=max_parallelism,
                    model_kwargs={"seed": seed},
                )
            ],
        )

    def __call__(self):
        return GenerationStrategy(steps=self._steps)


class GPEIStrategy(GenerationStrategy):
    def __init__(
        self,
        warmup_trials=128,
        warmup_parallelism=16,
        num_trials=16,
        max_parallelism=1,
        seed=292288111,
    ):
        super().__init__(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=warmup_trials,
                    max_parallelism=warmup_parallelism,
                    model_kwargs={"seed": seed},
                ),
                GenerationStep(
                    model=Models.GPEI,
                    num_trials=num_trials,
                    max_parallelism=max_parallelism,
                ),
            ],
        )

    def __call__(self):
        return GenerationStrategy(steps=self._steps)


class ExplicitGPEIStrategy(GenerationStrategy):
    ...



if __name__ == "__main__":
    strat = GridSearchStrategy(seed=4)
    print(type(strat()))

# gs = GenerationStrategy(
#     [
#         GenerationStep(
#             model=Models.SOBOL,
#             num_trials=128,
#             max_parallelism=16,
#             model_kwargs={"seed": 1},
#         ),
#         GenerationStep(
#             model=Models.GPEI,
#             num_trials=16,
#             max_parallelism=1,
#         )
# GenerationStep(
#     model=Models.BOTORCH_MODULAR,
#     num_trials=16,
#     max_parallelism=1,
#     model_kwargs={
#         "surrogate": Surrogate(
#             # botorch_model_class=SingleTaskGP,
#             # botorch_model_class=LocalizationGP,
#             botorch_model_class=ConstrainedLocalizationGP,
#             mll_class=ExactMarginalLogLikelihood,
#         ),
#         "botorch_acqf_class": ExpectedImprovement,
#     },
#     model_gen_kwargs={
#         "model_gen_options": {
#             "optimizer_kwargs": {
#                 "num_restarts": 40,
#                 "raw_samples": 1024,
#             }
#         }
#     },
# ),
#     ]
# )
