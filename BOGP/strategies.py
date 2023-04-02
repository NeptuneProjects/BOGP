#!/usr/bin/env python3

from typing import Any, Protocol

class Strategy(Protocol):
    def configure(self) -> None:
        ...

class StrategyFactory(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Strategy:
        ...

