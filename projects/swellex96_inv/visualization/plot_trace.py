#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def main():
    data = np.load("/Users/williamjenkins/Research/Projects/BOGP/data/swellex96_S5_VLA_inv/outputs/runs/ei_20-10_0590.npz")
    print(data["Y"])

    X = data["X"]
    Y = data["Y"]
    
    plt.plot(X[:, 0])
    plt.show()

    # plt.plot(np.minimum.accumulate(Y))
    # plt.show()

if __name__ == "__main__":
    main()
