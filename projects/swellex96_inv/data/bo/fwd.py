#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from obj import get_objective


def main(simulate: bool = False) -> None:
    objective = get_objective(simulate=simulate)
    if simulate:
        return objective(common.TRUE_SIM_VALUES)
    return objective(common.TRUE_EXP_VALUES)


if __name__ == "__main__":
    main()
