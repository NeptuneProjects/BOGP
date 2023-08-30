# -*- coding: utf-8 -*-

from typing import Protocol

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from oao.strategy import GridStrategy


class StrategyInterface(Protocol):
    def __call__(self):
        ...


class GridSearchStrategy(GridStrategy):
    def __init__(self, num_trials=12, max_parallelism=1, *args, **kwargs):
        super().__init__(
            num_trials=num_trials,
            max_parallelism=max_parallelism,
        )

    def __call__(self, *args, **kwargs):
        return self


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
