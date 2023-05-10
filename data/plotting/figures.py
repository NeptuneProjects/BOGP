#!/usr/bin/env python3

from argparse import ArgumentParser
from copy import deepcopy

# import ast
from pathlib import Path

import sys
import warnings

from ax.service.ax_client import AxClient
import matplotlib as mpl
from matplotlib import ticker
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tritonoa.io.profile import read_ssp
from tritonoa.at.models.kraken.runner import run_kraken
from tritonoa.sp.beamforming import beamformer
from tritonoa.sp.mfp import MatchedFieldProcessor

sys.path.insert(0, str(Path(__file__).parents[2]))
from conf.swellex96.optimization.common import SWELLEX96Paths
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
            # continue


def figure1():
    return show_sampling_density()


def figure2():
    # Reserved for GP model selection
    return


def figure3():
    return plot_training_1D()


def figure4():
    return plot_environment()


def figure5():
    return simulations_localization()


def figure6():
    # Reserved to show n_warmup tradeoff curves
    return


def figure7():
    # Reserved for showing CPU time
    return


def figure8a():
    return experimental_localization()


def figure8b():
    print("Hello!")
    # Reserved for error analysis
    return

def figure10():
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


def experimental_posterior():
    AMBSURF_PATH = (
        SWELLEX96Paths.ambiguity_surfaces / "148-166-201-235-283-338-388_200x100"
    )
    # TIMESTEP = 310
    TIMESTEP = 100
    STRATEGY = "gpei"
    CBAR_KW = {"location": "top", "pad": 0}
    CONTOUR_KW = {
        "origin": "lower",
        "extent": [0, 10, 50, 75],
    }
    LABEL_KW = {"ha": "center", "va": "center", "rotation": 90, "fontsize": "large"}
    SCATTER_KW = {
        "marker": "o",
        "facecolors": "none",
        "edgecolors": "white",
        "linewidth": 0.1,
        # "alpha": 1,
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
    XLIM = [0, 10]

    results = (
        SWELLEX96Paths.outputs
        / "localization"
        / "experimental"
        / "serial_001"
        / "src_z=60.00__tilt=-1.00__time_step=100__rec_r=5.08"
        / "gpei"
        / "292288111"
        / "client.json"
    )
    client = AxClient(verbose_logging=False).load_from_json_file(results)
    
    return
    rvec_f = np.linspace(0.01, 10, 500)
    zvec_f = np.linspace(1, 200, 100)
    f = np.load(AMBSURF_PATH / f"ambsurf_mf_t={TIMESTEP}.npy")
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

    # for i in range(nrows):
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

    axrow = axs[i]

    # Objective function
    ax = axrow[0]
    vmin = 0
    vmax = 0.6
    vmid = vmax / 2
    if i == 0:
        ax.set_title("Objective function $f(\mathbf{X})$", **TITLE_KW)
    im = ax.contourf(
        rvec_f, zvec_f, f, levels=np.linspace(vmin, vmax, NLEVELS), **CONTOUR_KW
    )
    ax.scatter(x, y, **SCATTER_KW)
    ax.scatter(src_r, src_z, **SOURCE_KW)
    ax.scatter(src_r_est, src_z_est, **SOURCE_EST_KW)
    ax.invert_yaxis()
    if i == 0:
        ax.set_xticklabels([])
        ax.set_ylim([75, 50])
    if i == 1:
        ax.set_xlabel("Range [km]")
        ax.set_ylim([200, 0])
    ax.set_xlim(XLIM)
    ax.set_ylabel("Depth [m]")
    fig.colorbar(im, ax=ax, ticks=[vmin, vmid, vmax], **CBAR_KW)
    if i == 0:
        label = "Depth-constrained\nsearch space"
    else:
        label = "Full search space"
    ax.text(-0.27, 0.5, label, transform=ax.transAxes, **LABEL_KW)

    # Mean function
    ax = axrow[1]
    if i == 0:
        ax.set_title("Mean function $\mu(\mathbf{X})$", **TITLE_KW)
    im = ax.contourf(
        rvec, zvec, mean["z"], levels=np.linspace(vmin, vmax, NLEVELS), **CONTOUR_KW
    )
    ax.scatter(x, y, **SCATTER_KW)
    ax.scatter(src_r, src_z, **SOURCE_KW)
    ax.scatter(src_r_est, src_z_est, **SOURCE_EST_KW)
    ax.invert_yaxis()
    ax.set_xlim(XLIM)
    ax.set_xticklabels([])
    if i == 0:
        ax.set_ylim([75, 50])
    else:
        ax.set_ylim([200, 0])
    ax.set_yticklabels([])
    fig.colorbar(im, ax=ax, ticks=[vmin, vmid, vmax], **CBAR_KW)

    # Covariance function
    ax = axrow[2]
    vmin = 0
    vmax = 0.08
    vmid = vmax / 2
    if i == 0:
        ax.set_title("Covar. function $2\sigma(\mathbf{X})$", **TITLE_KW)
    im = ax.contourf(
        rvec, zvec, se["z"], levels=np.linspace(vmin, vmax, NLEVELS), **CONTOUR_KW
    )
    ax.scatter(x, y, **SCATTER_KW)
    ax.scatter(src_r, src_z, **SOURCE_KW)
    ax.scatter(src_r_est, src_z_est, **SOURCE_EST_KW)
    ax.invert_yaxis()
    ax.set_xlim(XLIM)
    ax.set_xticklabels([])
    if i == 0:
        ax.set_ylim([75, 50])
    else:
        ax.set_ylim([200, 0])
    ax.set_yticklabels([])
    fig.colorbar(im, ax=ax, ticks=[vmin, vmid, vmax], **CBAR_KW)

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

    ax.plot(X_test, alpha, color="tab:red", label="$\\alpha(\mathbf{X})$")
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
        figsize=(4, 5),
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
        xytext=(800, 185),
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
        450,
        "Mudrock layer\n$\\rho = 2.06\ \mathrm{g\ cm^{-3}}$\n$a=0.06\ \mathrm{dB \ km^{-1}\ Hz^{-1}}$",
        bbox=props,
        va="center",
    )
    ax.annotate(
        "Bedrock halfspace\n$c = 5200\ \mathrm{m\ s^{-1}}$\n$\\rho = 2.66\ \mathrm{g\ cm^{-3}}$\n$a=0.02\ \mathrm{dB \ km^{-1}\ Hz^{-1}}$",
        xy=(1500, 1090),
        xycoords="data",
        xytext=(1460, 940),
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


def plot_experimental_results(df):
    STRATEGY_KEY = [
        "High-res MFP",
        "Sobol",
        "Grid",
        "SBL",
        "Sobol+GP/EI",
        "",
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

    ax.plot(X_test, y_actual, color="tab:green", label="$f(\mathbf{X})$")
    ax.plot(X_test, mean, label="$\mu(\mathbf{X})$")
    ax.fill_between(
        X_test.squeeze(), lcb, ucb, alpha=0.25, label="$\pm2\sigma(\mathbf{X})$"
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
        "font.size": 10,
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

    trials_to_plot = [0, 1, 2, 10, 20, 30, 50, 60, 70]

    fig = plt.figure(figsize=(6, 12))
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
        ax.set_ylabel("$f(\mathbf{X})$", rotation=0, ha="center", va="center")

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
        ax.set_ylabel("$\\alpha(\mathbf{X})$", rotation=0, ha="center", va="center")

    del handles[4], labels[4]
    handles[-1] = Line2D([0], [0], color="r", marker="x", linestyle=":")

    handles2, labels2 = ax.get_legend_handles_labels()
    ax.legend(
        handles + handles2,
        labels + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -1.0),
        ncols=4,
    )

    ax.set_xlabel("$\mathbf{X}\ (R_{src}$ [km])")

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
    print(results_dir)

    if not SMOKE_TEST:
        df = pd.read_csv(results_dir / "aggregated_results.csv")

    rec_r = 5.0
    src_z = 60.0
    strategies = {
        "grid": "Grid Search (144)",
        "sobol": "Sobol Sequence (144)",
        "gpei": "Sobol (128) + GP-EI (16)",
    }
    seed = int("002406475")

    params = {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["cm"],
        "font.size": 10,
    }
    mpl.rcParams.update(params)

    # sns.set_theme(style="darkgrid")
    fig = plt.figure(figsize=(12, 4))
    outer_gs = gridspec.GridSpec(nrows=1, ncols=3, wspace=0.16)
    margin = 1.1

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
        ax.scatter(
            x=df[selection]["rec_r"],
            y=df[selection]["src_z"],
            s=25,
            alpha=0.5,
            color="k",
            edgecolors="none",
        )
        ax.set_xlim(rec_r - margin * 1, rec_r + margin * 1)
        ax.set_ylim(src_z - margin * 40, src_z + margin * 40)
        ax.invert_yaxis()

        ax_histx = fig.add_subplot(inner_gs[0, 0], sharex=ax)
        ax_histx.hist(df[selection]["rec_r"], bins=100, color="k")
        ax_histx.set_ylim(0, 13)
        plt.setp(ax_histx.get_xticklabels(), visible=False)
        ax_histx.set_title(strat_name.upper())

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
            {"grid": "Grid Search", "sobol": "Sobol Sequence", "gpei": "GP-EI"}
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


# def simulations_range_est():
#     SMOKE_TEST = False

#     sim_dir = ROOT / "Data" / "range_estimation" / "simulation"
#     serial = "serial_230217"
#     if not SMOKE_TEST:
#         df = pd.read_csv(sim_dir / serial / "results" / "collated.csv")

#     RANGES = [1.0, 3.0, 5.0, 7.0]
#     TITLE_KW = {"ha": "left", "va": "top", "x": 0}

#     fig, axs = plt.subplots(
#         figsize=(12, 6), nrows=4, ncols=3, gridspec_kw={"wspace": 0.15}
#     )

#     # Column 1 - Objective Function ============================================
#     # TODO: Read in high-res objective function
#     TITLE = "Ambiguity surface: $f(\mathbf{x})$"
#     XLIM = [0, 10]
#     XLABEL = "Range [km]"
#     YLIM = [0, 1.2]

#     axcol = axs[:, 0]
#     # Set ranges
#     [
#         axcol[i].text(
#             -0.52,
#             0.5,
#             f"$R_\mathrm{{src}} = {r}$ km",
#             transform=axcol[i].transAxes,
#             fontsize="large",
#             ha="left",
#         )
#         for i, r in enumerate(RANGES)
#     ]

#     for ax, r in zip(axcol, RANGES):
#         # fname = (
#         #     ROOT
#         #     / "Data"
#         #     / "range_estimation"
#         #     / "simulation"
#         #     / "serial_230217"
#         #     / f"rec_r={r:.1f}__src_z=60.0__snr=20"
#         #     / "grid"
#         #     / "seed_0002406475"
#         #     / "results.json"
#         # )

#         if not SMOKE_TEST:
#             selection = (
#                 (df["seed"] == int("0002406475"))
#                 & (df["strategy"] == "Grid")
#                 & (df["range"] == r)
#             )
#             df_obj = df[selection].sort_values("rec_r")
#             g = sns.lineplot(data=df_obj, x="rec_r", y="bartlett", ax=ax)
#             g.set(xlabel=None, ylabel=None)
#             ax.axvline(r, color="r")

#     # Set x axis
#     [axcol[i].set_xlim(XLIM) for i in range(len(RANGES))]
#     [axcol[i].set_xticklabels([]) for i in range(len(RANGES) - 1)]
#     # [axcol[-1].set_xlabel(None) for i in range(len(RANGES) - 1)]
#     [axcol[-1].set_xlabel(XLABEL)]

#     # Set y axis
#     [axcol[i].set_ylim(YLIM) for i, _ in enumerate(RANGES)]

#     # Set title
#     axcol[0].set_title(TITLE, **TITLE_KW)

#     # Column 2 - Performance History ===========================================
#     TITLE = "Best observed: $\hat{f}(\mathbf{x})$"
#     XLIM = [0, 201]
#     XLABEL = "Evaluation"
#     YLIM = [0, 1.2]

#     axcol = axs[:, 1]

#     count = 0
#     for ax, r in zip(axcol, RANGES):
#         if not SMOKE_TEST:
#             selection = df["range"] == r
#             g = sns.lineplot(
#                 data=df[selection],
#                 x="trial_index",
#                 y="best_values",
#                 hue="strategy",
#                 ax=ax,
#                 legend="auto" if count == 3 else None,
#             )
#             g.set(xlabel=None, ylabel=None)
#             if count == 3:
#                 sns.move_legend(
#                     ax,
#                     "upper center",
#                     bbox_to_anchor=(1.1, -0.5),
#                     ncol=7,
#                     title="Strategy",
#                 )
#             count += 1

#     # Set x axis
#     [axcol[i].set_xlim(XLIM) for i in range(len(RANGES))]
#     [axcol[i].set_xticklabels([]) for i in range(len(RANGES) - 1)]
#     [axcol[-1].set_xlabel(XLABEL)]

#     # Set y axis
#     [axcol[i].set_ylim(YLIM) for i in range(len(RANGES))]

#     # Set title
#     axcol[0].set_title(TITLE, **TITLE_KW)

#     # Column 3 - Error History =================================================
#     TITLE = "Error history: $\\vert\hat{R}_{src} - R_{src}\\vert$ [km]"
#     YLIM = [0, 8]

#     axcol = axs[:, 2]

#     for ax, r in zip(axcol, RANGES):
#         if not SMOKE_TEST:
#             selection = df["range"] == r
#             g = sns.lineplot(
#                 data=df[selection],
#                 x="trial_index",
#                 y="best_range_error",
#                 hue="strategy",
#                 ax=ax,
#                 legend=None,
#             )
#             g.set(xlabel=None, ylabel=None)

#     # Set x axis
#     [axcol[i].set_xlim(XLIM) for i in range(len(RANGES))]
#     [axcol[i].set_xticklabels([]) for i in range(len(RANGES) - 1)]
#     [axcol[-1].set_xlabel(XLABEL)]

#     # Set y axis
#     [axcol[i].set_ylim(YLIM) for i in range(len(RANGES))]

#     # Set title
#     axcol[0].set_title(TITLE, **TITLE_KW)

#     return fig


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--figures", type=str, default="10")
    args = parser.parse_args()
    # figures = list(map(lambda i: int(i.strip()), args.figures.split(",")))
    figures = list(map(lambda i: i.strip(), args.figures.split(",")))
    main(figures)
