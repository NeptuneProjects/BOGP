#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
from obj import get_objective


params = {
    "rec_r": 1.07,
    "src_z": 72.04,
    "tilt": 2.14,
    "h_w": 217.71,
    "h_s": 20.55,
    "c_p_sed_top": 1573.32,
    "dc_p_sed": 16.96,
}


def main(simulate: bool = False) -> None:
    objective = get_objective(simulate=simulate)
    if simulate:
        return objective(params)
    return objective(common.TRUE_EXP_VALUES)


if __name__ == "__main__":
    print(main(simulate=True))
