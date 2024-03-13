#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
from enum import Enum
import itertools
from typing import Any, Optional
import warnings


class UnequalLengthWarning(Warning):
    pass


class DropOptions(Enum):
    HEAD = "head"
    TAIL = "tail"


@dataclass
class IndexedParameterization:
    scenario: dict[str, list] = field(default_factory=dict)

    def __len__(self) -> int:
        if not self.equal_length:
            warnings.warn("Not all lists have same length!", UnequalLengthWarning)
        else:
            return len(next(iter(self.scenario.values())))

    @property
    def equal_length(self) -> bool:
        return all(
            len(l) == len(next(iter(self.scenario.values())))
            for l in self.scenario.values()
        )

    def parameterize(self) -> list[dict]:
        if not self.scenario:
            return [{}]
        if not self.equal_length:
            warnings.warn(
                "Not all lists have same length! Use equalize_length() to "
                "fix this. Values with index greater than the minimum length "
                "have been dropped.",
                UnequalLengthWarning,
            )
        return [
            dict(zip(self.scenario.keys(), values))
            for values in zip(*self.scenario.values())
        ]

    def equalize_length(self, drop: Optional[str] = "tail") -> None:
        dropopt = DropOptions(drop)

        if not self.equal_length:
            min_length = min(len(l) for l in self.scenario.values())
            if dropopt == DropOptions.HEAD:
                self.scenario = {k: v[-min_length:] for k, v in self.scenario.items()}
            elif dropopt == DropOptions.TAIL:
                self.scenario = {k: v[:min_length] for k, v in self.scenario.items()}


@dataclass
class PermutedParameterization:
    scenario: dict[list] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(list(itertools.product(*self.scenario.values())))

    def parameterize(self) -> list[dict]:
        permutations = list(itertools.product(*self.scenario.values()))
        return [dict(zip(self.scenario.keys(), p)) for p in permutations]


@dataclass
class Parameterization:
    indexed: IndexedParameterization = field(default_factory=IndexedParameterization)
    permuted: PermutedParameterization = field(default_factory=PermutedParameterization)
    fixed: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.scenarios = self.permute_parameterizations()
        self.scenarios = self.add_fixed_parameterization()

    def __getitem__(self, index: int) -> dict:
        return self.scenarios[index]

    def __len__(self) -> int:
        return len(self.scenarios)

    def permute_parameterizations(self) -> list[dict]:
        indexed = self.indexed.parameterize()
        permuted = self.permuted.parameterize()
        scenarios = []
        for item in indexed:
            for perm in permuted:
                scenarios.append({**item, **perm})
        return scenarios

    def add_fixed_parameterization(self) -> list[dict]:
        return [{**self.fixed, **s} for s in self.scenarios]


def main():
    indexed = {
        "timestep": list(range(50, 60)),
        "range": list(range(5, 15)),
    }
    permuted = {"depths": [50, 60, 70], "snr": [0, 10]}
    fixed = {"name": "test"}

    idx_param = IndexedParameterization(indexed)
    perm_param = PermutedParameterization(permuted)

    param = Parameterization(indexed=idx_param, permuted=perm_param, fixed=fixed)
    print(param[5])


if __name__ == "__main__":
    main()
