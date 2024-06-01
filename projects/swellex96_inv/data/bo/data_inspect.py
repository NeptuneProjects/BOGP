#!/usr/bin/env python3

from pathlib import Path

import numpy as np

import common, helpers, obj

# PATH = Path.cwd().parent / "data/swellex96_S5_VLA_inv/outputs/runs/sim_logei/logei_100-32_0083.npz"
PATH = (
    Path.cwd().parent
    / "data/swellex96_S5_VLA_inv/outputs/runs/sim_random/random_10000-32_9216.npz"
)


def main():
    data = np.load(PATH)
    X = data["X"]
    Y = data["Y"]
    best = helpers.get_best_params(X, Y, common.SEARCH_SPACE)
    params = {
        k: v
        for k, v in zip(
            [d["name"] for d in common.SEARCH_SPACE], best.squeeze().tolist()
        )
    }

    objective = obj.get_objective(simulate=True)
    out = objective(params)
    print(params)
    print(Y.min())
    print(out)


if __name__ == "__main__":
    main()
