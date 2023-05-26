#!/usr/bin/env python3

from argparse import ArgumentParser
from copy import deepcopy

# import ast
from pathlib import Path
import string

import sys
import warnings

from ax.service.ax_client import AxClient
import jax.numpy as jnp
import matplotlib as mpl
from matplotlib import ticker
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tinygp import GaussianProcess, kernels
from tritonoa.io.profile import read_ssp
from tritonoa.at.models.kraken.runner import run_kraken
from tritonoa.sp.beamforming import beamformer
from tritonoa.sp.mfp import MatchedFieldProcessor

sys.path.insert(0, str(Path(__file__).parents[2]))
from conf.swellex96.optimization.common import FREQ, SWELLEX96Paths
from data.collate import load_mfp_results, NO_DATA
import optimization.utils as utils

ROOT = Path(__file__).parents[3]
FIGURE_PATH = ROOT / "reports" / "manuscripts" / "JASA" / "figures"
SAVEFIG_KWARGS = {"dpi": 200, "facecolor": "white", "bbox_inches": "tight"}


def main(figures: list):
    for figure in figures:
        print(f"Producing Figure {figure} " + 60 * "-")
        try:
            fig = eval(f"figure{figure}()")
            fig.savefig(FIGURE_PATH / f"figure{figure}.pdf", **SAVEFIG_KWARGS)
        except NameError:
            warnings.warn(f"Figure {figure} is not implemented yet.")
            raise NotImplementedError(f"Figure {figure} is not implemented yet.")


def figure1():
    return show_sampling_density()


def figure2():
    return plot_model_selection()


def figure3():
    return plot_training_1D()


def figure4():
    return plot_environment()


def figure5():
    return simulations_localization()


def figure6():
    return plot_run_times()


def figure7():
    return experimental_localization()


def figure8():
    return experimental_error()


def figure9():
    return experimental_posterior()


def figure999():
    plot_training_2D()


def set_rcparams():
    params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["cm"],
        "font.size": 10,
    }
    mpl.rcParams.update(params)


def adjust_subplotticklabels(ax, low=None, high=None):
    ticklabels = ax.get_yticklabels()
    if low is not None:
        ticklabels[low].set_va("bottom")
    if high is not None:
        ticklabels[high].set_va("top")


def initialize_model_selection():
    environment = utils.load_env_from_json(SWELLEX96Paths.environment_data) | {
        "tmpdir": "."
    }
    freq = FREQ
    rec_r_true = 5.0
    src_z_true = 60.0
    true_parameters = {
        "rec_r": rec_r_true,
        "src_z": src_z_true,
    }

    K = utils.simulate_covariance(
        runner=run_kraken,
        parameters=environment | true_parameters,
        freq=freq,
    )

    MFP = MatchedFieldProcessor(
        runner=run_kraken,
        covariance_matrix=K,
        freq=freq,
        parameters=environment | {"src_z": src_z_true},
        beamformer=beamformer,
    )
    num_rvec = 200
    dr = 1.0
    rec_r_lim = (rec_r_true - dr, rec_r_true + dr)
    sigma_f = 0.1
    x_test = np.linspace(rec_r_lim[0], rec_r_lim[1], num_rvec)
    y_true = MFP({"rec_r": x_test})

    n_samples = 50
    random = np.random.default_rng(1)
    x = random.uniform(rec_r_lim[0], rec_r_lim[1], size=n_samples)
    y = MFP({"rec_r": x})

    return x, y, x_test, y_true, sigma_f


def neg_log_likelihood(X, y, sigma_f, length_scale, sigma_y):
    gp = build_gp(X, jnp.exp(length_scale), sigma_f, jnp.exp(sigma_y))
    return -gp.log_probability(y)


def build_gp(x, length_scale, sigma_f, sigma_y):
    kernel = (sigma_f**2) * kernels.ExpSquared(scale=length_scale)
    # kernel = (sigma_f ** 2) * kernels.Matern52(scale=length_scale)
    return GaussianProcess(kernel, x, diag=sigma_y**2)


def plot_gp_pred(x, y, xtest, sigma_f, length_scale, sigma_y, ax=None):
    if ax is None:
        ax = plt.gca()
    gp = build_gp(x, length_scale, sigma_f, sigma_y)
    cond_gp = gp.condition(y, xtest).gp
    mu, var = cond_gp.loc, cond_gp.variance
    ax.scatter(x, y, s=50, c="k", marker="x", label="Data")
    ax.plot(xtest, mu, color="k", label="Mean")
    ax.fill_between(
        xtest,
        mu + 2 * jnp.sqrt(var),
        mu - 2 * jnp.sqrt(var),
        color="k",
        alpha=0.3,
        edgecolor="none",
        label="Confidence",
    )
    sns.despine()
    return ax


def plot_marginal_likelihood_surface(
    x, y, sigma_f, l_space, sigma_y_space, levels=None, ax=None
):
    if ax is None:
        ax = plt.gca()
    P = jnp.stack(jnp.meshgrid(l_space, sigma_y_space), axis=0)
    Z = jnp.apply_along_axis(lambda p: neg_log_likelihood(x, y, sigma_f, *p), 0, P)
    Z = Z - Z.min()
    Z = Z.at[jnp.where(Z == 0)].set(0.01)
    im = ax.contourf(
        *jnp.exp(P),
        Z,
        levels,
        cmap="bone_r",
        locator=ticker.LogLocator(numticks=levels),
    )

    ax.contour(
        *jnp.exp(P),
        Z,
        levels,
        colors="k",
        linewidths=0.5,
        locator=ticker.LogLocator(numticks=levels),
    )
    return P, Z, im


def add_subfigure_labels(axs, x=0.5, y=-0.2):
    [
        ax.text(
            x,
            y,
            f"({string.ascii_lowercase[i]})",
            ha="center",
            va="top",
            transform=ax.transAxes,
            size=14,
        )
        for i, ax in enumerate(axs)
    ]


def plot_model_selection():
    set_rcparams()
    x, y, x_test, y_true, sigma_f = initialize_model_selection()

    ngrid = 100
    levels = 20
    length_space = jnp.linspace(jnp.log(1.0e-2), jnp.log(1.0e0), ngrid)
    sigma_y_space = jnp.linspace(jnp.log(1.0e-4), jnp.log(1.0e-1), ngrid)

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))

    ax = axs[0]
    P, Z, im = plot_marginal_likelihood_surface(
        x, y, sigma_f, length_space, sigma_y_space, levels, ax=ax
    )
    fig.colorbar(im, ax=ax)
    ind_x, ind_y = jnp.unravel_index(jnp.argmin(Z), Z.shape)

    params_optim = [jnp.exp(P[0])[ind_x, ind_y], jnp.exp(P[1])[ind_x, ind_y]]
    params_suboptim = [5e-1, 5e-2]

    ax.scatter(marker="*", color="w", s=100, zorder=50, edgecolor="k", *params_optim)
    ax.text(
        4e-2,
        params_optim[1],
        "(b)",
        ha="right",
        va="center",
        bbox=dict(alpha=0.85, facecolor="w", linewidth=0, pad=0.5),
    )
    ax.scatter(color="w", marker="*", s=100, zorder=50, edgecolor="k", *params_suboptim)
    ax.text(
        4e-1,
        params_suboptim[1],
        "(c)",
        ha="right",
        va="center",
        bbox=dict(alpha=0.85, facecolor="w", linewidth=0, pad=0.5),
    )
    ax.set_xlabel("Length scale $l$ [km]")
    ax.set_ylabel("Noise std. dev. $\sigma_y$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(
        r"$-\log\left[p(\mathbf{y} \vert \mathbf{X}, l, \sigma_y)\right]$ (Normalized)"
    )
    sns.despine()

    ax = axs[1]
    ax = plot_gp(x, y, x_test, y_true, sigma_f, params_optim, ax=ax)
    ax.set_xlabel("Range [km]")
    ax.set_ylabel("$f(\mathbf{x})$")
    ax.legend(loc="upper right", bbox_to_anchor=(1.1, 1.0))
    title = rf"$l={{{format_sci_notation(f'{params_optim[0]:.2E}')}}}$, $\sigma_y={{{format_sci_notation(f'{params_optim[1]:.2E}')}}}$"
    ax.set_title(title)

    ax = axs[2]
    ax = plot_gp(x, y, x_test, y_true, sigma_f, params_suboptim, ax=ax)
    ax.set_xlabel("Range [km]")
    ax.set_ylabel("$f(\mathbf{x})$")
    # ax.legend()
    title = rf"$l={{{format_sci_notation(f'{params_suboptim[0]:.2E}')}}}$, $\sigma_y={{{format_sci_notation(f'{params_suboptim[1]:.2E}')}}}$"
    ax.set_title(title)
    add_subfigure_labels(axs, x=0.5, y=-0.2)

    return fig


def format_sci_notation(float_str):
    base, exponent = float_str.split("E")
    return rf"{base} \times 10^{{{int(exponent)}}}"


def plot_gp(x, y, x_test, y_true, sigma_f, params, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(x_test, y_true, linestyle="--", c="k", label="True")
    ax = plot_gp_pred(x, y, x_test, sigma_f, *params, ax=ax)
    ax.set_ylim(0, 1.1)
    return ax


def experimental_error():
    serial = "serial_001"
    fname = (
        SWELLEX96Paths.outputs
        / "localization"
        / "experimental"
        / serial
        / "results"
        / "collated_results.csv"
    )
    df = pd.read_csv(fname)
    return plot_experimental_error(df)


def experimental_localization():
    serial = "serial_001"
    fname = (
        SWELLEX96Paths.outputs
        / "localization"
        / "experimental"
        / serial
        / "results"
        / "collated_results.csv"
    )
    df = pd.read_csv(fname)
    return plot_experimental_results(df)


def get_true_parameters_from(scenario):
    true_rec_r = float(scenario.split("rec_r=")[1].split("_")[0])
    true_src_z = float(scenario.split("src_z=")[1].split("_")[0])
    return true_rec_r, true_src_z


def experimental_posterior():
    AMBSURF_PATH = (
        SWELLEX96Paths.ambiguity_surfaces / "148-166-201-235-283-338-388_200x100"
    )
    TIMESTEP = 200
    STRATEGY = "gpei"
    set_rcparams()
    CBAR_KW = {"location": "top", "pad": 0}
    CONTOUR_KW = {
        "origin": "lower",
        "extent": [0, 10, 50, 75],
    }
    LABEL_KW = {"ha": "center", "va": "center", "rotation": 90, "fontsize": "large"}
    SCATTER_KW = {
        "marker": "o",
        "facecolors": "tab:orange",
        # "edgecolors": "white",
        # "linewidth": 0.1,
        "alpha": 1,
        "s": 5,
        "zorder": 40,
    }
    SOURCE_KW = {
        "marker": "s",
        "facecolors": "none",
        "edgecolors": "lime",
        "linewidth": 1,
        "s": 40,
        "zorder": 50,
    }
    SOURCE_EST_KW = {
        "marker": "s",
        "facecolors": "none",
        "edgecolors": "r",
        "linewidth": 1,
        "s": 20,
        "zorder": 60,
    }
    TITLE_KW = {"y": 1.2, "fontsize": "large"}
    NLEVELS = 21

    scenario = list(
        (SWELLEX96Paths.outputs / "localization" / "experimental" / "serial_001").glob(
            f"*time_step={TIMESTEP}*"
        )
    )[0]

    results = (
        SWELLEX96Paths.outputs
        / "localization"
        / "experimental"
        / "serial_001"
        / scenario.name
        / STRATEGY
        / "292288111"
        / "client.json"
    )
    client = AxClient(verbose_logging=False).load_from_json_file(results)

    true_rec_r, true_src_z = get_true_parameters_from(scenario.name)

    rvec_f = np.linspace(0.1, 8, 200)
    zvec_f = np.linspace(1, 200, 100)
    f = np.load(AMBSURF_PATH / f"surface_{TIMESTEP}.npy")
    src_z_ind, src_r_ind = np.unravel_index(np.argmax(f), (len(zvec_f), len(rvec_f)))
    src_r = rvec_f[src_r_ind]
    src_z = zvec_f[src_z_ind]

    nrows = 1
    ncols = 3
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(12, 3),
        gridspec_kw={"wspace": 0.1, "hspace": 0.1},
    )
    client.fit_model()
    data = client.get_contour_plot()
    best, _ = client.get_best_parameters()
    src_r_est = best["rec_r"]
    src_z_est = best["src_z"]
    rvec = data.data["data"][0]["x"]
    zvec = data.data["data"][0]["y"]
    mean = data.data["data"][0]
    se = data.data["data"][1]
    x = data.data["data"][3]["x"]
    y = data.data["data"][3]["y"]

    XLIM = [true_rec_r - 1.0, true_rec_r + 1.0]
    YLIM = [true_src_z + 40.0, true_src_z - 40.0]

    # Objective function
    ax = axs[0]
    vmin = 0
    vmax = 1.0
    vmid = vmax / 2
    ax.set_title("Objective function $f(\mathbf{x})$", **TITLE_KW)
    im = ax.contourf(
        rvec_f, zvec_f, f, levels=np.linspace(vmin, vmax, NLEVELS), **CONTOUR_KW
    )
    ax.scatter(x, y, **SCATTER_KW)
    ax.scatter(src_r, src_z, **SOURCE_KW)
    ax.scatter(src_r_est, src_z_est, **SOURCE_EST_KW)
    ax.set_xlim(XLIM)
    ax.set_xlabel("Range [km]")
    ax.invert_yaxis()
    ax.set_ylim(YLIM)
    ax.set_ylabel("Depth [m]")
    fig.colorbar(im, ax=ax, ticks=[vmin, vmid, vmax], **CBAR_KW)

    # Mean function
    ax = axs[1]
    ax.set_title("Mean $\mu(\mathbf{x})$", **TITLE_KW)
    im = ax.contourf(
        rvec, zvec, mean["z"], levels=np.linspace(vmin, vmax, NLEVELS), **CONTOUR_KW
    )
    ax.scatter(x, y, **SCATTER_KW)
    ax.scatter(src_r, src_z, **SOURCE_KW)
    ax.scatter(src_r_est, src_z_est, **SOURCE_EST_KW)
    ax.invert_yaxis()
    ax.set_xlim(XLIM)
    ax.set_xticklabels([])
    ax.set_ylim(YLIM)
    ax.set_yticklabels([])
    fig.colorbar(im, ax=ax, ticks=[vmin, vmid, vmax], **CBAR_KW)

    # Inset mean function
    axins = ax.inset_axes([0.05, 0.45, 0.5, 0.5], zorder=100)
    im = axins.contourf(
        rvec, zvec, mean["z"], levels=np.linspace(vmin, vmax, NLEVELS), **CONTOUR_KW
    )
    axins.scatter(x, y, s=10, c="tab:orange")
    axins.scatter(src_r, src_z, s=10, c="lime", marker="s")
    axins.scatter(src_r_est, src_z_est, s=15, c="red", marker="*")
    axins.invert_yaxis()
    xlim_ins = [src_r - 0.1, src_r + 0.1]
    ylim_ins = [src_z + 5, src_z - 5]
    axins.set_xlim(xlim_ins)
    axins.set_ylim(ylim_ins)
    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    _, connector_lines = ax.indicate_inset_zoom(axins, edgecolor="w")
    [con.set_visible(False) for con in connector_lines]

    for i in range(2):
        xy_inset = (i, i)
        xy_parent = (xlim_ins[i], ylim_ins[i])
        connect = mpatches.ConnectionPatch(
            xy_inset,
            xy_parent,
            "axes fraction",
            "data",
            color="w",
            alpha=0.5,
            axesA=axins,
            axesB=ax,
        )
        ax.add_patch(connect)

    # Covariance function
    ax = axs[2]
    vmin = 0
    vmax = 0.08
    vmid = vmax / 2
    ax.set_title("Standard error $2\sigma(\mathbf{x})$", **TITLE_KW)
    im = ax.contourf(
        rvec, zvec, se["z"], levels=np.linspace(vmin, vmax, NLEVELS), **CONTOUR_KW
    )
    ax.scatter(x, y, **SCATTER_KW)
    ax.scatter(src_r, src_z, **SOURCE_KW)
    ax.scatter(src_r_est, src_z_est, **SOURCE_EST_KW)
    ax.invert_yaxis()
    ax.set_xlim(XLIM)
    ax.set_xticklabels([])
    ax.set_ylim(YLIM)
    ax.set_yticklabels([])
    fig.colorbar(im, ax=ax, ticks=[vmin, vmid, vmax], **CBAR_KW)

    add_subfigure_labels(axs, x=0.5, y=-0.25)

    return fig


def experimental_range_est():
    AMBSURF_PATH = (
        ROOT
        / "Data"
        / "SWELLEX96"
        / "VLA"
        / "selected"
        / "multifreq"
        / "148.0-166.0-201.0-235.0-283.0-338.0-388.0"
    )
    # Load high-res MFP
    timesteps, ranges, depths = load_mfp_results(AMBSURF_PATH)

    # GPS Range
    gps_fname = ROOT / "Data" / "SWELLEX96" / "VLA" / "selected" / "gps_range.csv"
    df_gps = pd.read_csv(gps_fname, index_col=0)

    serial = "serial_constrained_50-75"
    fname = (
        ROOT
        / "Data"
        / "localization"
        / "experimental"
        / serial
        / "results"
        / "collated.csv"
    )
    df = pd.read_csv(fname, index_col=0)
    return plot_experimental_results(df, df_gps, timesteps, ranges, depths)


def format_error(error, est_timesteps):
    error_plot = np.empty(350)
    error_plot[:] = np.nan
    error_plot[est_timesteps] = error
    return error_plot


def get_candidates(alpha, alpha_prev=None):
    if alpha is None:
        max_alpha = None
    else:
        max_alpha = np.argmax(alpha)
        alpha /= alpha.max()

    if alpha_prev is None:
        max_alpha_prev = None
    else:
        max_alpha_prev = np.argmax(alpha_prev)
        alpha_prev /= alpha_prev.max()
    return max_alpha, max_alpha_prev


def load_training_data(loadpath):
    X_test = np.load(loadpath / "X_test.npy")
    y_actual = np.load(loadpath / "y_actual.npy")
    X_train = np.load(loadpath / "X_train.npy")
    y_train = np.load(loadpath / "y_train.npy")
    mean = np.load(loadpath / "mean.npy")
    lcb = np.load(loadpath / "lcb.npy")
    ucb = np.load(loadpath / "ucb.npy")
    alpha = np.load(loadpath / "alpha.npy")
    return X_test, y_actual, X_train, y_train, mean, lcb, ucb, alpha


def plot_acqf_1D(X_test, alpha, alpha_prev=None, ax=None):
    if ax is None:
        ax = plt.gca()

    max_alpha, max_alpha_prev = get_candidates(alpha, alpha_prev)

    ax.plot(X_test, alpha, color="tab:red", label="$\\alpha(\mathbf{x})$")
    ax.axvline(X_test[max_alpha], color="k", linestyle="-")
    if max_alpha_prev:
        ax.axvline(X_test[max_alpha_prev], color="r", linestyle=":")
    return ax


def plot_ambiguity_surface(
    B,
    rvec,
    zvec,
    ax=None,
    cmap="viridis",
    vmin=-10,
    vmax=0,
    interpolation="none",
    marker="*",
    markersize=15,
    markeredgewidth=1.5,
    markeredgecolor="k",
    markerfacecolor="w",
):
    if ax is None:
        ax = plt.gca()

    src_z_ind, src_r_ind = np.unravel_index(np.argmax(B), (len(zvec), len(rvec)))

    im = ax.imshow(
        B,
        aspect="auto",
        extent=[min(rvec), max(rvec), min(zvec), max(zvec)],
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        interpolation=interpolation,
        cmap=cmap,
    )
    ax.plot(
        rvec[src_r_ind],
        zvec[src_z_ind],
        marker=marker,
        markersize=markersize,
        markeredgewidth=markeredgewidth,
        markeredgecolor=markeredgecolor,
        markerfacecolor=markerfacecolor,
    )
    ax.invert_yaxis()
    return ax, im


def plot_environment():
    params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["cm"],
        "font.size": 10,
    }
    mpl.rcParams.update(params)
    zw, cw, _ = read_ssp(
        ROOT / "data" / "swellex96_S5_VLA" / "ctd" / "i9606.prn", 0, 3, header=None
    )
    zw = np.append(zw, 217)
    cw = np.append(cw, cw[-1])
    zb1 = np.array([np.NaN, 217, 240])
    cb1 = np.array([np.NaN, 1572.37, 1593.0])
    zb2 = np.array([np.NaN, 240, 1040])
    cb2 = np.array([np.NaN, 1881, 3245])
    z1 = np.concatenate([zw, zb1])
    c1 = np.concatenate([cw, cb1])

    upper_zlim = [240, 0]
    lower_zlim = [1100, 240]
    lower_clim = [1450, 1600]
    upper_clim = [1800, 3400]

    fig, axs = plt.subplots(
        figsize=(4, 4),
        nrows=2,
        ncols=2,
        gridspec_kw={"wspace": 0.05, "hspace": 0, "width_ratios": [0.67, 0.33]},
    )

    ax = axs[0, 0]
    ax.plot(c1, z1)
    ax.axhline(217, c="k", lw=1)
    ax.fill_between(lower_clim, 0, 217, color="lightblue", alpha=0.15, linewidth=0)
    ax.fill_between(lower_clim, 217, 240, color="yellow", alpha=0.15, linewidth=0)
    ax.set_xlim(lower_clim)
    ax.set_xlabel("Sound Speed [m/s]")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.invert_yaxis()
    ax.set_ylim(upper_zlim)
    ax.set_yticks([0, 50, 100, 150, 200, 217])
    ax.set_ylabel("Depth [m]", y=0)
    ax.spines.right.set_linestyle((0, (5, 10)))
    ax.spines.right.set_linewidth(0.5)
    ax.spines.bottom.set_visible(False)

    ax = axs[0, 1]
    ax.axhline(217, c="k", lw=1)
    ax.fill_between(upper_clim, 0, 217, color="lightblue", alpha=0.15, linewidth=0)
    ax.fill_between(upper_clim, 217, 240, color="yellow", alpha=0.15, linewidth=0)
    props = dict(boxstyle="round", facecolor="white", alpha=0.9)
    ax.annotate(
        "Sediment layer\n$\\rho = 1.76\ \mathrm{g\ cm^{-3}}$\n$a=0.2\ \mathrm{dB \ km^{-1}\ Hz^{-1}}$",
        xy=(2500, 235),
        xycoords="data",
        xytext=(800, 150),
        textcoords="data",
        bbox=props,
        arrowprops=dict(arrowstyle="->"),
    )
    ax.set_xlim(upper_clim)
    ax.xaxis.set_tick_params(top=True, bottom=False)
    ax.set_xticklabels([])
    ax.set_ylim(upper_zlim)
    ax.set_yticks([])
    ax.spines.left.set_linestyle((0, (5, 10)))
    ax.spines.left.set_linewidth(0.5)
    ax.spines.bottom.set_visible(False)

    ax = axs[1, 0]
    ax.axhline(242, c="k", lw=1)
    ax.axhline(1040, c="k", lw=1)
    ax.fill_between(lower_clim, 240, 1040, color="tan", alpha=0.15, linewidth=0)
    ax.fill_between(lower_clim, 1040, 1100, color="gray", alpha=0.15, linewidth=0)
    ax.text(
        1460,
        400,
        "Mudrock layer\n$\\rho = 2.06\ \mathrm{g\ cm^{-3}}$\n$a=0.06\ \mathrm{dB \ km^{-1}\ Hz^{-1}}$",
        bbox=props,
        va="center",
    )
    ax.annotate(
        "Bedrock halfspace\n$c = 5200\ \mathrm{m\ s^{-1}}$\n$\\rho = 2.66\ \mathrm{g\ cm^{-3}}$\n$a=0.02\ \mathrm{dB \ km^{-1}\ Hz^{-1}}$",
        xy=(1500, 1100),
        xycoords="data",
        xytext=(1460, 900),
        textcoords="data",
        bbox=props,
        arrowprops=dict(arrowstyle="->"),
    )
    ax.set_xlim(lower_clim)
    ax.set_xticklabels([])
    ax.set_ylim(lower_zlim)
    ax.set_yticks([240, 400, 600, 800, 1040])
    ax.spines.right.set_linestyle((0, (5, 10)))
    ax.spines.right.set_linewidth(0.5)
    ax.spines.top.set_visible(False)

    ax = axs[1, 1]
    ax.plot(cb2, zb2)
    ax.axhline(242, c="k", lw=1)
    ax.axhline(1040, c="k", lw=1)
    ax.fill_between(upper_clim, 240, 1040, color="tan", alpha=0.15, linewidth=0)
    ax.fill_between(upper_clim, 1040, 1100, color="gray", alpha=0.15, linewidth=0)
    ax.set_xlim(upper_clim)
    ax.invert_yaxis()
    ax.set_ylim(lower_zlim)
    ax.set_yticks([])
    ax.spines.left.set_linestyle((0, (5, 10)))
    ax.spines.left.set_linewidth(0.5)
    ax.spines.top.set_visible(False)

    return fig


def plot_experimental_error(df):
    STRATEGY_KEY = [
        "High-res MFP",
        "Sobol",
        "Grid",
        "SBL",
        "Sobol+GP/EI",
        "Sobol+GP/qEI",
    ]
    ERROR_KW = {"edgecolor": None, "s": 10}
    NO_DATA_KW = {"color": "black", "alpha": 0.25, "linewidth": 0, "label": None}
    no_data = NO_DATA
    XLIM = [0, 351]
    XTICKS = list(range(0, 351, 50))
    YLIM_R = [-1, 1]
    YTICKS_R = [-1, 0, 1]
    YLIM_Z = [-40, 40]
    YTICKS_Z = [-40, 0, 40]

    df_mfp = df[df["strategy"] == STRATEGY_KEY.pop(0)]

    fig = plt.figure(figsize=(12, 8), facecolor="white")
    sns.set_theme(style="darkgrid")
    set_rcparams()

    nrows, ncols = 5, 2
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.4, wspace=0.1)

    for i, strategy in enumerate(STRATEGY_KEY):
        # try:
        selection = df["strategy"] == strategy
        df_selection = df[selection].copy()
        df_selection["Range Error [km]"] = (
            df_selection["best_rec_r"].values - df_mfp["best_rec_r"].values
        )
        df_selection["Depth Error [m]"] = (
            df_selection["best_src_z"].values - df_mfp["best_src_z"].values
        )
        # except ValueError:
        #     selection = df["strategy"] == STRATEGY_KEY[-2]
        #     df_selection = df[selection].copy()
        #     df_selection["Range Error [km]"] = (
        #         df_selection["best_rec_r"].values - df_mfp["best_rec_r"].values
        #     )
        #     df_selection["Depth Error [m]"] = (
        #         df_selection["best_src_z"].values - df_mfp["best_src_z"].values
        #     )

        ax1 = fig.add_subplot(gs[i, 0])
        [ax1.axvspan(l[0] - 1, l[-1] + 1, zorder=5, **NO_DATA_KW) for l in no_data]
        sns.scatterplot(
            data=df_selection, x="Time Step", y="Range Error [km]", ax=ax1, **ERROR_KW
        )
        ax1.set_xlim(XLIM)
        ax1.set_xticks(XTICKS)
        ax1.set_ylim(YLIM_R)
        ax1.set_yticks(YTICKS_R)
        ax1.set_ylabel(strategy)
        ax1.tick_params(length=0)
        adjust_subplotticklabels(ax1, 0, -1)

        ax2 = fig.add_subplot(gs[i, 1])
        [ax2.axvspan(l[0] - 1, l[-1] + 1, zorder=5, **NO_DATA_KW) for l in no_data]
        sns.scatterplot(
            data=df_selection,
            x="Time Step",
            y="Depth Error [m]",
            ax=ax2,
            color="green",
            **ERROR_KW,
        )
        ax2.set_xlim(XLIM)
        ax2.set_xticks(XTICKS)
        ax2.set_ylim(YLIM_Z)
        ax2.set_yticks(YTICKS_Z)
        ax2.set_ylabel(None)
        ax2.tick_params(length=0)
        adjust_subplotticklabels(ax2, 0, -1)

        if i == 0:
            ax1.set_title("Range Error [km]")
            ax2.set_title("Depth Error [m]")
        if i == nrows - 1:
            ax1.set_xlabel("Time Step")
            ax2.set_xlabel(None)
        else:
            ax1.set_xlabel(None)
            ax2.set_xlabel(None)

    return fig


def plot_experimental_results(df):
    STRATEGY_KEY = [
        "High-res MFP",
        "Sobol",
        "Grid",
        "SBL",
        "Sobol+GP/EI",
        "Sobol+GP/qEI",
    ]
    no_data = NO_DATA
    XLIM = [0, 351]
    XTICKS = list(range(0, 351, 50))
    YLIM_R = [0, 10]
    YTICKS_R = [0, 5, 10]
    YLIM_Z = [100, 20]
    YTICKS_Z = [20, 60, 100]
    GPS_KW = {"color": "k", "label": "GPS Range", "legend": None, "zorder": 15}
    RANGE_KW = {
        "x": "Time Step",
        "y": "best_rec_r",
        "s": 10,
        "label": "Strategy",
        "legend": None,
        "alpha": 1.0,
        "linewidth": 0,
        "zorder": 20,
    }
    DEPTH_KW = {
        "x": "Time Step",
        "y": "best_src_z",
        "s": 10,
        "color": "green",
        "label": "Strategy",
        "legend": None,
        "linewidth": 0,
        "zorder": 20,
    }
    NO_DATA_KW = {"color": "black", "alpha": 0.25, "linewidth": 0, "label": None}

    fig = plt.figure(figsize=(12, 6), facecolor="white")
    sns.set_theme(style="darkgrid")
    set_rcparams()

    nrows, ncols = 3, 2
    gs = gridspec.GridSpec(nrows, ncols, figure=fig, hspace=0.4, wspace=0.1)

    k = 0
    for i in range(nrows):
        for j in range(ncols):
            strategy = STRATEGY_KEY[k]

            sgs = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=gs[i, j], hspace=0.1
            )

            # Range ====================================================
            ax1 = fig.add_subplot(sgs[0])

            [ax1.axvspan(l[0] - 1, l[-1] + 1, zorder=5, **NO_DATA_KW) for l in no_data]
            sns.lineplot(data=df, x="Time Step", y="Range [km]", ax=ax1, **GPS_KW)
            sns.scatterplot(data=df[df["strategy"] == strategy], ax=ax1, **RANGE_KW)
            ax1.set_xlim(XLIM)
            ax1.set_xticks(XTICKS)
            ax1.set_xticklabels([])
            ax1.set_xlabel(None)
            ax1.set_ylim(YLIM_R)
            ax1.set_yticks(YTICKS_R)
            ax1.tick_params(length=0)
            adjust_subplotticklabels(ax1, 0, -1)
            ax1.set_title(strategy)

            # Depth ====================================================
            ax2 = fig.add_subplot(sgs[1])

            [ax2.axvspan(l[0] - 1, l[-1] + 1, zorder=1, **NO_DATA_KW) for l in no_data]
            sns.scatterplot(data=df[df["strategy"] == strategy], ax=ax2, **DEPTH_KW)
            ax2.set_xlim(XLIM)
            ax1.set_xticks(XTICKS)
            ax2.set_ylim(YLIM_Z)
            ax2.set_yticks(YTICKS_Z)
            ax2.set_ylabel("Depth [m]")
            ax2.tick_params(length=0)
            adjust_subplotticklabels(ax2, -1, 0)

            if k == 4:
                ax1.set_ylabel("Range [km]", loc="bottom")
                ax2.set_xlabel("Time Step")
                ax2.set_ylabel("Depth [m]", loc="top")
            else:
                ax1.set_ylabel(None)
                ax2.set_ylabel(None)
                ax2.set_xlabel(None)

            k += 1

    return fig


def plot_gp_1D(
    X_test,
    y_actual,
    X_train,
    y_train,
    mean,
    lcb,
    ucb,
    alpha=None,
    alpha_prev=None,
    ax=None,
):
    if ax is None:
        ax = plt.gca()

    max_alpha, max_alpha_prev = get_candidates(alpha, alpha_prev)

    ax.plot(X_test, y_actual, color="tab:green", label="$f(\mathbf{x})$")
    ax.plot(X_test, mean, label="$\mu(\mathbf{x})$")
    ax.fill_between(
        X_test.squeeze(), lcb, ucb, alpha=0.25, label="$\pm2\sigma(\mathbf{x})$"
    )
    if not max_alpha_prev:
        ax.scatter(X_train, y_train, c="k", marker="x", label="Samples", zorder=40)
    else:
        ax.scatter(
            X_train[:-1], y_train[:-1], c="k", marker="x", label="Samples", zorder=40
        )
        ax.scatter(
            X_train[-1], y_train[-1], c="r", marker="x", label="Samples", zorder=50
        )
    if max_alpha is not None:
        ax.axvline(
            X_test[max_alpha], color="k", linestyle="-", label="Next sample $t+1$"
        )
    if max_alpha_prev is not None:
        ax.axvline(
            X_test[max_alpha_prev], color="r", linestyle=":", label="Current sample $t$"
        )

    return ax


def plot_training_1D():
    params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["cm"],
        "font.size": 14,
    }
    mpl.rcParams.update(params)
    loadpath = (
        ROOT / "data" / "swellex96_S5_VLA" / "outputs" / "range_estimation" / "demo"
    )
    X_test, y_actual, X_train, y_train, mean, lcb, ucb, alpha = load_training_data(
        loadpath
    )
    # Condition acquisition function to correct numerical issues:
    alpha[alpha < 0] = 0

    num_rand = X_train.shape[0] - mean.shape[0]
    xlim = [0, 10]
    ylim = [-0.1, 1.1]

    trials_to_plot = [0, 50, 51, 80]

    fig = plt.figure(figsize=(8, 12))
    outer_grid = fig.add_gridspec(len(trials_to_plot), 1, hspace=0.1)

    for i, trial in enumerate(trials_to_plot):
        inner_grid = outer_grid[i].subgridspec(2, 1, hspace=0, height_ratios=[3, 2])
        axs = inner_grid.subplots()

        ax = axs[0]
        ax = plot_gp_1D(
            X_test,
            y_actual,
            X_train[0 : num_rand + trial],
            y_train[0 : num_rand + trial],
            mean[trial],
            lcb[trial],
            ucb[trial],
            alpha[trial],
            alpha[trial - 1] if trial != 0 else None,
            ax=ax,
        )
        if trial == trials_to_plot[-1]:
            handles, labels = ax.get_legend_handles_labels()
        if i == 0:
            ax.text(0.1, 0.9, "Initialization", va="top")
        else:
            ax.text(0.1, 0.9, f"Trial {trial}", va="top")

        ax.set_xticklabels([])
        ax.set_xlim(xlim)
        ax.set_xlabel(None)
        ax.set_ylim(ylim)
        ax.set_ylabel("$f(\mathbf{x})$", rotation=0, ha="right", va="center")

        ax = axs[1]
        ax = plot_acqf_1D(
            X_test, alpha[trial], alpha[trial - 1] if trial != 0 else None, ax=ax
        )
        ax.set_xlim(xlim)
        if trial != trials_to_plot[-1]:
            ax.set_xticklabels([])
        ax.set_ylim(ylim)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("$\\alpha(\mathbf{x})$", rotation=0, ha="center", va="center")

    del handles[4], labels[4]
    handles[-1] = Line2D([0], [0], color="r", marker="x", linestyle=":")

    handles2, labels2 = ax.get_legend_handles_labels()
    ax.legend(
        handles + handles2,
        labels + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.7),
        ncols=4,
    )

    ax.set_xlabel("$\mathbf{x}\ (R_{src}$ [km])")

    return fig


def plot_training_2D():
    loadpath = ROOT / "Data" / "localization" / "demo"
    (
        X_test,
        y_actual,
        X_train,
        _,
        mean,
        _,
        ucb,
        alpha_array,
    ) = load_training_data(loadpath)
    # Condition acquisition function to correct numerical issues:
    alpha_array[alpha_array < 0] = 0
    IMSHOW_KW = {
        "aspect": "auto",
        "origin": "lower",
        "interpolation": "none",
        "extent": [0, 10, 0, 200],
        "vmin": 0,
        "vmax": 1,
    }
    ACTUAL_KW = {"marker": "s", "facecolors": "none", "edgecolors": "w", "zorder": 30}
    NEXT_KW = {"facecolors": "none", "edgecolors": "r", "zorder": 60}
    SCATTER_KW = {"c": "w", "marker": "o", "zorder": 40, "alpha": 0.5, "s": 1}
    M = len(np.unique(X_test[:, 0]))
    N = len(np.unique(X_test[:, 1]))

    max_f, _ = get_candidates(np.reshape(y_actual, (M, N)))
    max_f = np.unravel_index(max_f, y_actual.shape)

    num_rand = X_train.shape[0] - mean.shape[0]

    trials_to_plot = [0, 749]

    fig = plt.figure(figsize=(12, 6))
    outer_grid = fig.add_gridspec(len(trials_to_plot), 1, hspace=0.15)

    for i, trial in enumerate(trials_to_plot):
        alpha = alpha_array[trial]
        alpha_prev = alpha_array[trial - 1] if trial != 0 else None

        max_alpha, max_alpha_prev = get_candidates(
            np.reshape(alpha, (M, N)),
            np.reshape(alpha_prev, (M, N)) if trial != 0 else None,
        )
        max_alpha = np.unravel_index(max_alpha, alpha.shape)
        if max_alpha_prev is not None:
            max_alpha_prev = np.unravel_index(max_alpha_prev, alpha_prev.shape)

        inner_grid = outer_grid[i].subgridspec(1, 4, wspace=0)
        axs = inner_grid.subplots()

        # True objective function (ambiguity surface)
        ax = axs[0]
        if i == 0:
            ax.set_title("Objective function $f(\mathbf{X})$")
        ax.imshow(np.reshape(y_actual, (M, N)), **IMSHOW_KW)
        ax.scatter(*X_test[max_f], **ACTUAL_KW)
        if not max_alpha_prev:
            ax.scatter(X_train[:, 0], X_train[:, 1], **SCATTER_KW)
        else:
            ax.scatter(X_train[:-1, 0], X_train[:-1, 1], **SCATTER_KW)
            ax.scatter(
                X_train[-1, 0],
                X_train[-1, 1],
                c="r",
                marker="x",
                label="Samples",
                zorder=50,
            )
        if max_alpha is not None:
            ax.scatter(*X_test[max_alpha], **NEXT_KW)
        ax.invert_yaxis()
        if trial == trials_to_plot[-1]:
            ax.set_xlabel("$R_{src}$ [km]")
            ax.set_ylabel("$z_{src}$ [m]")
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        # Mean function
        ax = axs[1]
        if i == 0:
            ax.set_title("Mean function $\mu(\mathbf{X})$")
        ax.imshow(np.reshape(mean[trial], (M, N)), **IMSHOW_KW)
        ax.scatter(*X_test[max_f], **ACTUAL_KW)
        if not max_alpha_prev:
            ax.scatter(X_train[:, 0], X_train[:, 1], **SCATTER_KW)
        else:
            ax.scatter(X_train[:-1, 0], X_train[:-1, 1], **SCATTER_KW)
            ax.scatter(
                X_train[-1, 0],
                X_train[-1, 1],
                c="r",
                marker="x",
                label="Samples",
                zorder=50,
            )
        if max_alpha is not None:
            ax.scatter(*X_test[max_alpha], **NEXT_KW)
        ax.invert_yaxis()
        ax.set_xticklabels([])
        ax.set_xlabel(None)
        ax.set_yticklabels([])
        ax.set_ylabel(None)

        # Covariance function
        ax = axs[2]
        if i == 0:
            ax.set_title("Covar. function $2\sigma(\mathbf{X})$")
        im = ax.imshow(np.reshape(ucb[trial], (M, N)), **IMSHOW_KW)
        if not max_alpha_prev:
            ax.scatter(X_train[:, 0], X_train[:, 1], **SCATTER_KW)
        else:
            ax.scatter(X_train[:-1, 0], X_train[:-1, 1], **SCATTER_KW)
            ax.scatter(
                X_train[-1, 0],
                X_train[-1, 1],
                c="r",
                marker="x",
                label="Samples",
                zorder=50,
            )
        if max_alpha is not None:
            ax.scatter(*X_test[max_alpha], **NEXT_KW)
        ax.invert_yaxis()
        ax.set_xticklabels([])
        ax.set_xlabel(None)
        ax.set_yticklabels([])
        ax.set_ylabel(None)

        # Colorbar
        if trial == trials_to_plot[-1]:
            cax = ax.inset_axes([-0.5, -0.25, 1.0, 0.15])
            fig.colorbar(im, ax=ax, cax=cax, orientation="horizontal")

        # Acquisition function
        ax = axs[3]
        if i == 0:
            ax.set_title("Acq. function $\log(\\alpha(\mathbf{X}))$")
        ACQ_IMSHOW_KW = deepcopy(IMSHOW_KW)
        ln = LogNorm(
            vmin=ACQ_IMSHOW_KW.pop("vmin") + 0.00001, vmax=ACQ_IMSHOW_KW.pop("vmax")
        )
        ax.imshow(np.reshape(alpha, (M, N)), norm=ln, **ACQ_IMSHOW_KW)
        LogNorm()
        if max_alpha is not None:
            ax.scatter(*X_test[max_alpha], **NEXT_KW)
        ax.invert_yaxis()
        ax.set_xticklabels([])
        ax.set_xlabel(None)
        ax.set_yticklabels([])
        ax.set_ylabel(None)

    return fig


def show_sampling_density():
    SMOKE_TEST = False
    serial = "serial_000"
    results_dir = (
        ROOT
        / "data"
        / "swellex96_S5_VLA"
        / "outputs"
        / "localization"
        / "simulation"
        / serial
        / "results"
    )

    if not SMOKE_TEST:
        df = pd.read_csv(results_dir / "aggregated_results.csv")

    rec_r = 5.0
    src_z = 60.0
    strategies = {
        "grid": "Grid Search (144 trials)",
        "sobol": "Sobol Sequence (144 trials)",
        "gpei": "Sobol (128 trials) + GP-EI (16 trials)",
    }
    seed = int("002406475")

    params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["cm"],
        "font.size": 10,
    }
    mpl.rcParams.update(params)

    AMBSURF_KW = {"vmin": 0, "vmax": 1, "marker": None}

    # sns.set_theme(style="darkgrid")
    fig = plt.figure(figsize=(12, 4))
    outer_gs = gridspec.GridSpec(nrows=1, ncols=3, wspace=0.16)
    margin = 1.1

    environment = utils.load_env_from_json(
        ROOT / "data" / "swellex96_S5_VLA" / "env_models" / "swellex96.json"
    )
    env_parameters = environment | {"tmpdir": "."}
    NUM_RVEC = 100
    NUM_ZVEC = 100
    true_parameters = {
        "rec_r": rec_r,
        "src_z": src_z,
    }
    rvec = np.linspace(rec_r - margin * 1.0, rec_r + margin * 1.0, NUM_RVEC)
    zvec = np.linspace(src_z - margin * 40, src_z + margin * 40, NUM_ZVEC)
    K = utils.simulate_covariance(
        runner=run_kraken,
        parameters=env_parameters | true_parameters,
        freq=FREQ,
    )

    MFP = MatchedFieldProcessor(
        runner=run_kraken,
        covariance_matrix=K,
        freq=FREQ,
        parameters=env_parameters,
        beamformer=beamformer,
    )

    amb_surf = np.zeros((len(zvec), len(rvec)))
    for zz, z in enumerate(zvec):
        amb_surf[zz, :] = MFP({"src_z": z, "rec_r": rvec})

    for i, (strat, strat_name) in enumerate(strategies.items()):
        selection = (
            (df["param_rec_r"] == rec_r)
            & (df["param_src_z"] == src_z)
            & (df["strategy"] == strat)
            & ((df["seed"] == seed) | (df["seed"] == 0))
        )

        inner_gs = outer_gs[i].subgridspec(
            nrows=2,
            ncols=2,
            wspace=0.1,
            hspace=0.1,
            height_ratios=[1, 6],
            width_ratios=[6, 1],
        )

        ax = fig.add_subplot(inner_gs[1, 0])

        ax, _ = plot_ambiguity_surface(amb_surf, rvec, zvec, ax=ax, **AMBSURF_KW)

        ax.scatter(
            x=df[selection]["rec_r"],
            y=df[selection]["src_z"],
            s=10,
            alpha=0.95,
            color="tab:orange",
            edgecolors="none",
        )
        ax.set_xlim(rec_r - margin * 1, rec_r + margin * 1)
        ax.set_ylim(src_z - margin * 40, src_z + margin * 40)
        ax.invert_yaxis()

        ax_histx = fig.add_subplot(inner_gs[0, 0], sharex=ax)
        ax_histx.hist(df[selection]["rec_r"], bins=100, color="k")
        ax_histx.set_ylim(0, 13)
        plt.setp(ax_histx.get_xticklabels(), visible=False)
        ax_histx.set_title(strat_name)

        ax_histy = fig.add_subplot(inner_gs[1, 1], sharey=ax)
        ax_histy.hist(
            df[selection]["src_z"], bins=100, color="k", orientation="horizontal"
        )
        ax_histy.set_xlim(0, 13)
        plt.setp(ax_histy.get_yticklabels(), visible=False)

        ax.scatter(
            x=rec_r,
            y=src_z,
            marker="o",
            s=75,
            color="red",
            facecolor="none",
            linewidths=2,
        )

        if i == 0:
            ax.set_xlabel("Range [km]")
            ax.set_ylabel("Depth [m]")
            ax_histx.set_ylabel("Count")

        ax.text(
            0.5,
            -0.2,
            f"({string.ascii_lowercase[i]})",
            ha="center",
            va="top",
            transform=ax.transAxes,
            size=14,
        )

    return fig


def plot_run_times():
    PAPER = True
    SMOKE_TEST = False
    sns.set_theme(style="darkgrid")
    params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["cm"],
        "font.size": 10 if PAPER else 16,
    }
    mpl.rcParams.update(params)

    serial = "serial_001"
    results_dir = (
        ROOT
        / "data"
        / "swellex96_S5_VLA"
        / "outputs"
        / "localization"
        / "simulation"
        / serial
        / "results"
    )

    if not SMOKE_TEST:
        df = pd.read_csv(results_dir / "aggregated_results.csv")
        df["strategy"] = df["strategy"].map(
            {
                "grid": "Grid Search",
                "sobol": "Sobol Sequence",
                "gpei": "GP-EI",
                "qgpei": "GP-qEI",
            }
        )

    fig = plt.figure(figsize=(4, 2))
    ax = plt.gca()

    strategies = {
        "Sobol+GP/EI \&\nSobol+GP/qEI": "tab:blue",
        "Sobol Sequence": "tab:orange",
        "Grid Search": "tab:green",
    }

    selected_df = df[(df["strategy"] == "GP-EI") | (df["strategy"] == "GP-qEI")]
    for seed in selected_df["seed"].unique():
        sns.lineplot(
            data=selected_df[selected_df["seed"] == seed],
            x="run_time",
            y="best_value",
            color="tab:blue",
            alpha=0.1,
            ax=ax,
        )

    selected_df = df[df["strategy"] == "Sobol Sequence"]
    for seed in selected_df["seed"].unique():
        sns.lineplot(
            data=selected_df[selected_df["seed"] == seed],
            x="run_time",
            y="best_value",
            color="tab:orange",
            alpha=0.1,
            ax=ax,
        )

    selected_df = df[df["strategy"] == "Grid Search"]
    sns.lineplot(
        data=selected_df, x="run_time", y="best_value", color="tab:green", ax=ax
    )
    ax.set_xlim(-5, 400)
    ax.set_ylim(0, 1.01)
    ax.tick_params(length=0)
    ax.set_xlabel("Run time [s]")
    ax.set_ylabel("$\hat{f}(\mathbf{x})$")

    legend_entries = [
        mpl.lines.Line2D([0], [0], color=color, lw=4)
        # mpl.lines.Line2D([0], [0], color="tab:orange", lw=4),
        # mpl.lines.Line2D([0], [0], color="tab:green", lw=4),
        # mpl.lines.Line2D([0], [0], color="tab:red", lw=4),
        for color in strategies.values()
    ]
    plt.legend(legend_entries, strategies.keys(), loc="lower right", facecolor="white")

    return fig


def simulations_localization():
    PAPER = False
    SMOKE_TEST = False
    params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["cm"],
        "font.size": 12 if PAPER else 16,
    }
    mpl.rcParams.update(params)

    serial = "serial_000"
    results_dir = (
        ROOT
        / "data"
        / "swellex96_S5_VLA"
        / "outputs"
        / "localization"
        / "simulation"
        / serial
        / "results"
    )

    if not SMOKE_TEST:
        df = pd.read_csv(results_dir / "aggregated_results.csv")
        df["strategy"] = df["strategy"].map(
            {
                "grid": "Grid Search",
                "sobol": "Sobol Sequence",
                "gpei": "Sobol+GP/EI",
                "qgpei": "Sobol+GP/qEI",
            }
        )

    FREQUENCIES = [148, 166, 201, 235, 283, 338, 388]
    RANGES = [1.0, 3.0, 5.0, 7.0]
    TITLE_KW = {"ha": "center", "va": "bottom", "y": 1.05}

    fig, axs = plt.subplots(
        figsize=(16 if PAPER else 24, 10),
        nrows=4,
        ncols=4,
        gridspec_kw={"wspace": 0.16, "hspace": 0.17},
    )

    # Column 1 - Objective Function ============================================
    TITLE = "Ambiguity surface: $f(\mathbf{x})$"
    XLIM = [0, 10]
    XLABEL = "Range [km]"
    YLIM = [100, 20]
    YLABEL = "Depth [m]"
    AMBSURF_KW = {
        "vmin": 0,
        "vmax": 1,
        "marker": None,
        # "markeredgecolor": "w",
        # "markerfacecolor": "none"
    }

    NUM_RVEC = 80
    NUM_ZVEC = 80
    zvec = np.linspace(20, 100, NUM_ZVEC)
    min_r, max_r = 0.01, 8.0
    r_bounds_rel = [-1.0, 1.0]
    environment = utils.load_env_from_json(
        ROOT / "data" / "swellex96_S5_VLA" / "env_models" / "swellex96.json"
    )
    env_parameters = environment | {"tmpdir": "."}

    axcol = axs[:, 0]
    # Set ranges
    [
        axcol[i].text(
            -0.3,
            0.5,
            f"$R_\mathrm{{src}} = {r}$ km",
            transform=axcol[i].transAxes,
            ha="center",
            va="center",
            fontsize="x-large",
            rotation=90,
        )
        for i, r in enumerate(RANGES)
    ]

    count = 0
    for ax, r in zip(axcol, RANGES):
        print(f"Plotting objective function for range {r} km...")
        r_lower, r_upper = utils.adjust_bounds(
            lower=r + r_bounds_rel[0],
            lower_bound=min_r,
            upper=r + r_bounds_rel[1],
            upper_bound=max_r,
        )
        rvec = np.linspace(r_lower, r_upper, NUM_RVEC)

        true_parameters = {
            "rec_r": r,
            "src_z": 60,
        }
        if not SMOKE_TEST:
            K = utils.simulate_covariance(
                runner=run_kraken,
                parameters=env_parameters | true_parameters,
                freq=FREQUENCIES,
            )

            MFP = MatchedFieldProcessor(
                runner=run_kraken,
                covariance_matrix=K,
                freq=FREQUENCIES,
                parameters=env_parameters,
                beamformer=beamformer,
            )

            amb_surf = np.zeros((len(zvec), len(rvec)))
            for zz, z in enumerate(zvec):
                amb_surf[zz, :] = MFP({"src_z": z, "rec_r": rvec})

            ax, im = plot_ambiguity_surface(amb_surf, rvec, zvec, ax=ax, **AMBSURF_KW)
        else:
            ax, im = plot_ambiguity_surface(
                np.random.randn(len(zvec), len(rvec)), rvec, zvec, ax=ax, **AMBSURF_KW
            )

        if count == 0:
            cax = ax.inset_axes([0, 1, 1.0, 0.15])
            cbar = fig.colorbar(im, ax=ax, cax=cax, orientation="horizontal")
            cbar.ax.xaxis.set_ticks_position("top")
            cbar.ax.xaxis.set_label_position("top")
            count += 1

    # Set x axis
    [axcol[i].set_xlim(*[r + b for b in r_bounds_rel]) for i, r in enumerate(RANGES)]
    # [axcol[i].set_xticklabels([]) for i in range(len(RANGES) - 1)]
    # [axcol[-1].set_xlabel(None) for i in range(len(RANGES) - 1)]
    [axcol[-1].set_xlabel(XLABEL)]

    # Set y axis
    [axcol[i].set_ylim(YLIM) for i, _ in enumerate(RANGES)]
    axcol[-1].set_ylabel(YLABEL)

    # Set title
    axcol[0].set_title(TITLE, **TITLE_KW | {"y": 1.3})

    # Column 2 - Performance History ===========================================
    TITLE = "Best observed: $\hat{f}(\mathbf{x})$"
    XLIM = [0, 145]
    XLABEL = "Trial"
    YLIM = [0, 1.0]

    axcol = axs[:, 1]

    for ax, r in zip(axcol, RANGES):
        print(f"Plotting performance history for range {r} km...")
        if not SMOKE_TEST:
            selection = df["param_rec_r"] == r
            g = sns.lineplot(
                data=df[selection],
                x="trial_index",
                y="best_value",
                hue="strategy",
                ax=ax,
                legend=None,
            )
            g.set(xlabel=None, ylabel=None)

    # Set x axis
    [axcol[i].set_xlim(XLIM) for i in range(len(RANGES))]
    [axcol[i].set_xticklabels([]) for i in range(len(RANGES) - 1)]
    [axcol[-1].set_xlabel(XLABEL)]

    # Set y axis
    [axcol[i].set_ylim(YLIM) for i in range(len(RANGES))]

    # Set title
    axcol[0].set_title(TITLE, **TITLE_KW)

    # Column 3 - Range Error History ===========================================
    TITLE = "Range error history:\n$\\vert\hat{R}_{src} - R_{src}\\vert$ [km]"
    YLIM = [0, 1.1]

    axcol = axs[:, 2]

    count = 0
    for ax, r in zip(axcol, RANGES):
        print(f"Plotting range error for range {r} km...")
        if not SMOKE_TEST:
            selection = df["param_rec_r"] == r
            g = sns.lineplot(
                data=df[selection],
                x="trial_index",
                y="best_error_rec_r",
                hue="strategy",
                ax=ax,
                legend="auto" if count == 3 else None,
            )
            g.set(xlabel=None, ylabel=None)
            if count == 3:
                sns.move_legend(
                    ax,
                    "upper center",
                    bbox_to_anchor=(0.5, -0.3),
                    ncol=7,
                    title="Strategy",
                )
            count += 1

    # Set x axis
    [axcol[i].set_xlim(XLIM) for i in range(len(RANGES))]
    [axcol[i].set_xticklabels([]) for i in range(len(RANGES) - 1)]
    [axcol[-1].set_xlabel(XLABEL)]

    # Set y axis
    [axcol[i].set_ylim(YLIM) for i in range(len(RANGES))]

    # Set title
    axcol[0].set_title(TITLE, **TITLE_KW)

    # Column 4 - Depth Error History ===========================================
    TITLE = "Depth error history:\n$\\vert\hat{z}_{src} - z_{src}\\vert$ [m]"
    YLIM = [0, 41]

    axcol = axs[:, 3]

    for ax, r in zip(axcol, RANGES):
        print(f"Plotting depth error for range {r} km...")
        if not SMOKE_TEST:
            selection = df["param_rec_r"] == r
            g = sns.lineplot(
                data=df[selection],
                x="trial_index",
                y="best_error_src_z",
                hue="strategy",
                ax=ax,
                legend=None,
            )
            g.set(xlabel=None, ylabel=None)

    # Set x axis
    [axcol[i].set_xlim(XLIM) for i in range(len(RANGES))]
    [axcol[i].set_xticklabels([]) for i in range(len(RANGES) - 1)]
    [axcol[-1].set_xlabel(XLABEL)]

    # Set y axis
    [axcol[i].set_ylim(YLIM) for i in range(len(RANGES))]

    # Set title
    axcol[0].set_title(TITLE, **TITLE_KW)

    return fig


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--figures", type=str, default="10")
    args = parser.parse_args()
    # figures = list(map(lambda i: int(i.strip()), args.figures.split(",")))
    figures = list(map(lambda i: i.strip(), args.figures.split(",")))
    main(figures)
