#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass, field
import itertools
from typing import Any


class UnequalLengthWarning(Warning):
    pass


@dataclass
class IndexedParameterization:
    scenario: dict[str, list] = field(default_factory=dict)

    def __len__(self) -> int:
        if not self.equal_length:
            raise UnequalLengthWarning("Not all lists have same length!")
        else:
            return len(next(iter(self.scenario.values())))

    @property
    def equal_length(self) -> bool:
        it = iter(self.scenario.values())
        length_of_first = len(next(it))
        if not all(len(l) == length_of_first for l in it):
            return False
        return True

    def parameterize(self) -> list[dict]:
        if not self.scenario:
            return [{}]
        if not self.equal_length:
            raise UnequalLengthWarning("Not all lists have same length!")

        return [{k: v[i] for k, v in self.scenario.items()} for i in range(len(self))]


@dataclass
class PermutatedParameterization:
    scenario: dict[list] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(list(itertools.product(*self.scenario.values())))

    def parameterize(self) -> list[dict]:
        permutations = list(itertools.product(*self.scenario.values()))
        return [dict(zip(self.scenario.keys(), p)) for p in permutations]


@dataclass
class Parameterization:
    indexed: IndexedParameterization = field(
        default_factory=IndexedParameterization
    )
    permutated: PermutatedParameterization = field(
        default_factory=PermutatedParameterization
    )
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
        permutated = self.permutated.parameterize()
        scenarios = []
        for item in indexed:
            for perm in permutated:
                scenarios.append({**item, **perm})
        return scenarios

    def add_fixed_parameterization(self) -> list[dict]:
        return [{**s, **self.fixed} for s in self.scenarios]


def main():
    indexed = {
        "timestep": list(range(50, 60)),
        "range": list(range(5,15)),
    }
    permuted = {"depths": [50, 60, 70], "snr": [0, 10]}
    fixed = {"name": "test"}


    idx_param = IndexedParameterization(indexed)
    perm_param = PermutatedParameterization(permuted)

    param = Parameterization(indexed=idx_param, permutated=perm_param, fixed=fixed)
    print(param[5])
    

if __name__ == "__main__":
    main()
