#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
from scipy.stats.qmc import Sobol

plt.style.use(["science", "ieee"])

def sample_comparison():
    # Define the number of samples to generate
    num_samples = 512

    # Generate random samples
    random_samples = np.random.rand(num_samples, 2)

    # Generate unscrambled Sobol samples
    sobol1 = Sobol(d=2, scramble=False)
    sobol_samples1 = sobol1.random(num_samples)

    # Generate scrambled Sobol samples
    sobol2 = Sobol(d=2, scramble=True)
    sobol_samples2 = sobol2.random(num_samples)

    # Plot the samples
    fig, axs = plt.subplots(1, 3, figsize=(5.125, 1.75), gridspec_kw={"wspace": 0.1})
    ax = axs[0]
    ax.scatter(random_samples[:, 0], random_samples[:, 1], s=1)
    ax.set_aspect("equal")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("(a) Uniform Random", loc="left")
    
    ax = axs[1]
    ax.scatter(sobol_samples1[:, 0], sobol_samples1[:, 1], s=1)
    ax.set_aspect("equal")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title("(b) Sobol Sequence", loc="left")

    ax = axs[2]
    ax.scatter(sobol_samples2[:, 0], sobol_samples2[:, 1], s=1)
    ax.set_aspect("equal")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title("(c) Scrambled Sobol", loc="left")
    
    return fig


if __name__ == '__main__':
    fig = sample_comparison()
    plt.show()
