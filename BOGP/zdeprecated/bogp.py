#!/usr/bin/env python3

from pathlib import Path
import sys

import numpy as np
np.random.seed(1234)
import matplotlib.pyplot as plt
from skopt.learning import GaussianProcessRegressor
from skopt import Optimizer
from skopt.plots import plot_gaussian_process
from skopt.space import Categorical, Real
from skopt.space.transformers import Normalize
from skopt.utils import use_named_args
from tqdm import tqdm


sys.path.insert(0, '/Users/williamjenkins/Research/Code/TritonOA/')
from tritonoa.kraken import KRAKENParameterization
from tritonoa.sp import ambiguity_function

import utils

noise_level = 1.e-8

# def objective(x, noise_level=noise_level):
#     return np.sin(5 * x[0] * (1 - np.tanh(x[0] ** 2)) +\
#         np.random.randn() * noise_level)

# def objective_wo_noise(x):
#     return objective(x, noise_level=0)

def plot_optimizer(res, n_iter, max_iters=5):
    if n_iter == 0:
        show_legend = True
    else:
        show_legend = False
    ax = plt.subplot(max_iters, 2, 2 * n_iter + 1)
    # Plot GP(x) + contours
    ax = plot_gaussian_process(res, ax=ax,
                               objective=objective,
                               noise_level=noise_level,
                               show_legend=show_legend, show_title=True,
                               show_next_point=False, show_acq_func=False)
    ax.set_ylabel("")
    ax.set_xlabel("")
    if n_iter < max_iters - 1:
        ax.get_xaxis().set_ticklabels([])
    # Plot EI(x)
    ax = plt.subplot(max_iters, 2, 2 * n_iter + 2)
    ax = plot_gaussian_process(res, ax=ax,
                               noise_level=noise_level,
                               show_legend=show_legend, show_title=False,
                               show_next_point=True, show_acq_func=True,
                               show_observations=False,
                               show_mu=False)
    ax.set_ylabel("")
    ax.set_xlabel("")
    if n_iter < max_iters - 1:
        ax.get_xaxis().set_ticklabels([])



dimensions = [
    Real(3, 6, transform="normalize", name="rec_r")
]

# Get data from simulated field:
data = np.load('/Users/williamjenkins/Research/Projects/BOGP/Data/Experiments/Localization2D/Simulated/Scene001/ReceivedData/data.npz')
K = data["K"]
# Get fixed parameters:
exppath = Path("/Users/williamjenkins/Research/Projects/BOGP/Data/Experiments/Localization2D/Simulated/Scene001/ReceivedData/parameters.json")
parameterization = KRAKENParameterization(path=exppath)
parameters = parameterization.parameters





@use_named_args(dimensions=dimensions)
def objective(**searchparams):
    gamma = 1e4
    for k, v in searchparams.items():
        parameters[k] = v

    parameterization = KRAKENParameterization(parameters=parameters)
    p_rep = parameterization.run()
    return 1 - gamma * ambiguity_function(K, p_rep, atype="bartlett").item()


# acq_func_kwargs = {"xi": 100000}
acq_func_kwargs = None


regressor = GaussianProcessRegressor(
    kernel=None,
    alpha=1e-10,
    optimizer="fmin_l_bfgs_b",
    n_restarts_optimizer=0,
    normalize_y=True,
    copy_X_train=True,
    random_state=2009,
    noise=noise_level
)


opt = Optimizer(
    dimensions,
    base_estimator="gp",
    n_initial_points=5,
    initial_point_generator="sobol",
    acq_func="EI",
    acq_optimizer="lbfgs",
    random_state=2009,
    n_jobs=1,
    acq_func_kwargs=acq_func_kwargs,
    acq_optimizer_kwargs=None
)


# xvec = np.linspace(3, 6, 100)
# B = np.zeros_like(xvec)
# for i, x in enumerate(xvec):
#     parameters["rec_r"] = x
#     parameterization = KRAKENParameterization(parameters=parameters)
#     p_rep = parameterization.run()
#     B[i] = 1 - ambiguity_function(K, p_rep, atype="bartlett").item()

# print(B.min(), B.max())
# plt.plot(xvec, B)
# plt.show()





plot_args = {
    "objective": objective,
    "noise_level": 0,
    "show_legend": True,
    "show_title": True,
    "show_next_point": True,
    "show_acq_func": True
}

for i in tqdm(range(100)):
    next_x = opt.ask()
    f_val = objective(next_x)
    opt.tell(next_x, f_val)

# fig = plt.figure()
# fig.suptitle("Standard GP kernel")
# for i in tqdm(range(10)):
#     next_x = opt.ask()
#     f_val = objective(next_x)
#     res = opt.tell(next_x, f_val)
#     if i >= 5:
#         plot_optimizer(res, n_iter=i-5, max_iters=5)
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()

print("Complete.")

results = opt.get_result()
plt.figure()
ax = plot_gaussian_process(results, **plot_args)
plt.show()
# path = f"/Users/williamjenkins/Desktop/Collector/{i:02d}.png"
print(results.models)