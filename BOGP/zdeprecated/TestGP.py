#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 16:14:59 2021

@author: William Jenkins
"""
import math

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pyro.contrib.gp as gp
# from scipy.stats import norm
import torch
import torch.autograd as autograd
from torch.distributions.normal import Normal
import torch.optim as optim
from torch.distributions import constraints, transform_to


def f(x):
    """Objective Function"""
    y = (4 * x - 1)**2 + torch.sin(8 * math.pi * x)
    return y


def find_a_candidate(gpmodel, acq_func, x_init, lower_bound=0, upper_bound=1):
    constraint = constraints.interval(lower_bound, upper_bound)
    unconstrained_x_init = transform_to(constraint).inv(x_init)
    unconstrained_x = unconstrained_x_init.clone().detach().requires_grad_(True)
    minimizer = optim.LBFGS([unconstrained_x], line_search_fn='strong_wolfe')
    
    def closure():
        minimizer.zero_grad()
        x = transform_to(constraint)(unconstrained_x)
        # alpha = lower_confidence_bound(gpmodel, x)
        alpha = acq_func_handler(acq_func, gpmodel, x)
        autograd.backward(unconstrained_x, autograd.grad(alpha, unconstrained_x))
        return alpha
    
    minimizer.step(closure)
    x_cand = transform_to(constraint)(unconstrained_x)
    return x_cand.detach()


def acq_func_handler(func, gpmodel, x, kappa=20, xi=0):
    if func == "lcb":
        alpha = lower_confidence_bound(gpmodel, x, kappa=kappa)
    elif func == "ei":
        alpha = expected_improvement(gpmodel, x, xi=xi)
    elif func == "pi":
        alpha = probability_of_improvement(gpmodel, x, xi=xi)
    return alpha


def lower_confidence_bound(gpmodel, x, kappa=20):
    mu, variance = gpmodel(x, full_cov=False, noiseless=False)
    sigma = variance.sqrt()
    return mu - kappa * sigma


def expected_improvement(gpmodel, x, xi=0):
    mu, variance = gpmodel(x, full_cov=False, noiseless=False)
    sigma = variance.sqrt()
    dist = Normal(mu, sigma)
    y_min = gpmodel.y.min()
    a = (mu - y_min - xi)
    z = a / sigma
    return a * dist.cdf(z) + sigma * dist.pdf(z)


def probability_of_improvement(gpmodel, x, xi=0):
    mu, variance = gpmodel(x, full_cov=False, noiseless=False)
    sigma = variance.sqrt()
    dist = Normal(mu, sigma)
    y_min = gpmodel.y.min()
    z = (mu - y_min - xi) / sigma
    return dist.cdf(z)
    # return norm.cdf(z)


def next_x(gpmodel, acq_func, lower_bound=0, upper_bound=1, num_candidates=5):
    candidates = []
    acq_values = []
    
    x_init = gpmodel.X[-1:]
    for _ in range(num_candidates):
        x_cand = find_a_candidate(gpmodel, acq_func, x_init, lower_bound, upper_bound)
        # alpha = lower_confidence_bound(gpmodel, x_cand)
        alpha = acq_func_handler(acq_func, gpmodel, x_cand)
        candidates.append(x_cand)
        acq_values.append(alpha)
        x_init = x_cand.new_empty(1).uniform_(lower_bound, upper_bound)
    
    argmin = torch.min(torch.cat(acq_values), dim=0)[1].item()
    return candidates[argmin]


def plot_objective_function(x, y, show=True, figsize=(4,3)):
    fig = plt.figure(figsize=figsize)
    plt.plot(x, y)
    plt.xlim(0, 1)
    plt.ylim(-2, 10)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.title(f'Objective Function')
    if show:
        plt.show()
    else:
        plt.close()
    return fig


def update_posterior(gpmodel, x_new):
    y = f(x_new)
    X = torch.cat([gpmodel.X, x_new])
    y = torch.cat([gpmodel.y, y])
    gpmodel.set_data(X, y)
    optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.001)
    gp.util.train(gpmodel, optimizer)
    return gpmodel, optimizer


def plot(gs, gpmodel, acq_func, xmin, xlabel=None, with_title=True, ylim1=None, ylim2=None):
    if xlabel is None:
        xlabel = "xmin"
    else:
        xlabel = f"x{xlabel}"
    
    Xnew = torch.linspace(0., 1., 101)
    ax1 = plt.subplot(gs[0])
    ax1.plot(gpmodel.X.detach().cpu().numpy(), gpmodel.y.detach().cpu().numpy(), "kx")
    with torch.no_grad():
        loc, var = gpmodel(Xnew, full_cov=False, noiseless=False)
        sd = var.sqrt()
        ax1.plot(Xnew.detach().cpu().numpy(), loc.detach().cpu().numpy(), "r", lw=2)
        ax1.fill_between(Xnew.detach().cpu().numpy(), (loc - 2*sd).detach().cpu().numpy(), (loc + 2*sd).detach().cpu().numpy(), color="C0", alpha=0.3)
    ax1.set_xlim(0., 1.)
    if ylim1 is not None:
        ax1.set_ylim(ylim1[0], ylim1[1])
    ax1.set_title(f"Find {xlabel}")
    if with_title:
        ax1.set_ylabel("GP Regression")
    
    ax2 = plt.subplot(gs[1])
    with torch.no_grad():

        # ax2.plot(Xnew.detach().cpu().numpy(), lower_confidence_bound(gpmodel, Xnew).detach().cpu().numpy())
        # ax2.plot(xmin.detach().cpu().numpy(), lower_confidence_bound(gpmodel, xmin).detach().cpu().numpy(), "^", markersize=10, label=f"{xlabel} = {xmin.item():.5f}")
        ax2.plot(Xnew.detach().cpu().numpy(), acq_func_handler(acq_func, gpmodel, Xnew).detach().cpu().numpy())
        ax2.plot(xmin.detach().cpu().numpy(), acq_func_handler(acq_func, gpmodel, xmin).detach().cpu().numpy(), "^", markersize=10, label=f"{xlabel} = {xmin.item():.5f}")
    ax2.set_xlim(0., 1.)
    if ylim2 is not None:
        ax2.set_ylim(ylim2[0], ylim2[1])
    if with_title:
        ax2.set_ylabel("Acquisition Function")
    ax2.set_xlabel('x')
    ax2.legend(loc=1)


def main(device):
    dtype = torch.double

    x = torch.linspace(0, 1, 101)
    y = f(x)

    _ = plot_objective_function(x, y, figsize=(8,3), show=False)

    X = torch.tensor([0.1, 0.8])
    y = f(X)

    gpmodel = gp.models.GPRegression(X, y, gp.kernels.Matern52(input_dim=1), noise=0 * torch.tensor(0.1), jitter=5.0e-4)

    optimizer = torch.optim.Adam(gpmodel.parameters(), lr=0.001)
    gp.util.train(gpmodel, optimizer)

    func_list = ["ei"]
    acq_func = func_list[0]

    for i in range(5):
        xmin = next_x(gpmodel, acq_func)

        fig = plt.figure(figsize=(6,7))
        gs = gridspec.GridSpec(2,1)
        plot(gs, gpmodel, acq_func, xmin, xlabel=i+1, with_title=True, ylim1=(-2, 10), ylim2=(-30, 10))
        plt.show()

        gpmodel, optimizer = update_posterior(gpmodel, xmin)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main(device)