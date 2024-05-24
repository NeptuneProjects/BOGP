#!/usr/bin/env python
# -*- coding: utf-8 -*-

import common
from obj import get_objective


def main():
    objective = get_objective(simulate=False)
    y = objective(common.TRUE_EXP_VALUES)
    print(y)


if __name__ == "__main__":
    main()
