#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def main(budget: int):
    names = ["BAxUS", "EI", "Sobol"]
    runs = [Y_baxus, Y_ei, Y_Sobol]
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, run in zip(names, runs):
        fx = np.maximum.accumulate(run.cpu())
        # plt.plot(-fx + branin.optimal_value, marker="", lw=3)
        plt.plot(1 - fx, marker="", lw=3)

    plt.ylabel("Regret", fontsize=18)
    plt.xlabel("Number of evaluations", fontsize=18)
    plt.title(f"Bartlett Power", fontsize=24)
    plt.xlim([0, budget])
    plt.yscale("log")

    plt.grid(True)
    plt.tight_layout()
    plt.legend(
        names + ["Global optimal value"],
        loc="lower center",
        bbox_to_anchor=(0, -0.08, 1, 1),
        bbox_transform=plt.gcf().transFigure,
        ncol=4,
        fontsize=16,
    )
    plt.show()