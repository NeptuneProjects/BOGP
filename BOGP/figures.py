#!/usr/bin/env python3

from argparse import ArgumentParser
from copy import deepcopy

# import ast
from pathlib import Path

# import sys
import warnings

# sys.path.insert(0, Path.cwd() / "Source")

from ax.service.ax_client import AxClient
from matplotlib import ticker
from matplotlib.colors import LogNorm
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from collate import get_error, load_mfp_results
from tritonoa.io import read_ssp

ROOT = Path.cwd()
FIGURE_PATH = ROOT / "Reports" / "JASA" / "figures"
SAVEFIG_KWARGS = {"dpi": 200, "facecolor": "white", "bbox_inches": "tight"}


def main(figures: list):
    for figure in figures:
        print(f"Producing Figure {figure:02d} " + 60 * "-")
        try:
            fig = eval(f"figure{figure}()")
            fig.savefig(FIGURE_PATH / f"figure{figure}.pdf", **SAVEFIG_KWARGS)
        except NameError:
            warnings.warn(f"Figure {figure} is not implemented yet.")
            raise NotImplementedError(f"Figure {figure} is not implemented yet.")
            # continue


def figure1():
    return plot_training_1D()


def figure2():
    return plot_training_2D()


def figure3():
    return plot_environment()


def figure4():
    return simulations_range_est()


# def figure5():
#     return simulations_localization()


def figure6():
    return experimental_range_est()


def figure7():
    return experimental_localization()


def figure8():
    return experimental_posterior()


def adjust_subplotticklabels(ax, low=None, high=None):
    ticklabels = ax.get_yticklabels()
    if low is not None:
        ticklabels[low].set_va("bottom")
    if high is not None:
        ticklabels[high].set_va("top")


def experimental_localization():
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

    serial = "serial_full_depth"
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
    return plot_experimental_results(
        df, df_gps, timesteps, ranges, depths, ylim_z=[200, 0], yticks_z=[0, 100, 200]
    )


def experimental_posterior():
    # TODO: Implement 2-D posterior view using MFP & Ax, e.g. at CPA
    return None


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
    color="w",
    markersize=15,
    markeredgewidth=1.5,
    markeredgecolor="k",
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
        color=color,
        markersize=markersize,
        markeredgewidth=markeredgewidth,
        markeredgecolor=markeredgecolor,
    )
    ax.invert_yaxis()
    return ax, im


def plot_environment():
    zw, cw, _ = read_ssp(
        ROOT / "Data" / "SWELLEX96" / "CTD" / "i9606.prn", 0, 3, header=None
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
        "Bedrock halfspace\n$c = 5200\ \mathrm{m\ s^{-1}}}$\n$\\rho = 2.66\ \mathrm{g\ cm^{-3}}$\n$a=0.02\ \mathrm{dB \ km^{-1}\ Hz^{-1}}$",
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


def plot_experimental_results(
    df, df_gps, timesteps, ranges, depths, ylim_z=[80, 40], yticks_z=[40, 60, 80]
):
    STRATEGY_KEY = [
        "High-res MFP",
        "Grid",
        "SBL",
        "LHS",
        "Rand",
        "Sobol",
        "PI",
        "EI",
        "qEI",
    ]
    no_data = [
        list(range(73, 85)),
        list(range(95, 103)),
        list(range(187, 199)),
        list(range(287, 294)),
        list(range(302, 309)),
    ]
    XLIM = [0, 351]
    XTICKS = list(range(0, 351, 50))
    YLIM_R = [0, 10]
    YTICKS_R = [0, 5, 10]
    YLIM_R_ERR = [1, 10000]
    YTICKS_R_ERR = [1, 100, 10000]
    YLIM_Z = [80, 40]
    YTICKS_Z = [40, 60, 80]
    YLIM_Z_ERR = [1e-2, 1e2]
    YTICKS_Z_ERR = [0.01, 1, 100]
    GPS_KW = {"color": "k", "label": "GPS Range", "legend": None, "zorder": 15}
    RANGE_KW = {
        "x": "Time Step",
        "y": "rec_r",
        "s": 10,
        "label": "Strategy",
        "legend": None,
        "alpha": 1.0,
        "linewidth": 0,
        "zorder": 20,
    }
    DEPTH_KW = {
        "x": "Time Step",
        "y": "src_z",
        "s": 10,
        "color": "green",
        "label": "Strategy",
        "legend": None,
        "linewidth": 0,
        "zorder": 20,
    }
    ERROR_KW = {
        "color": "tab:red",
        # "legend": None,
        "linewidth": 0.75,
        "alpha": 0.5,
        "zorder": 5,
    }
    NO_DATA_KW = {"color": "black", "alpha": 0.25, "linewidth": 0, "label": None}

    fig = plt.figure(figsize=(12, 8), facecolor="white")
    sns.set_theme(style="darkgrid")

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.25)

    k = 0
    for i in range(3):
        for j in range(3):
            strategy = STRATEGY_KEY[k]

            # Range ====================================================

            sgs = gridspec.GridSpecFromSubplotSpec(
                2, 1, subplot_spec=gs[i, j], hspace=0.05
            )

            # Range Error
            if k > 0:
                ax2 = fig.add_subplot(sgs[0])
                # ax2.grid(False)

                error, est_timesteps = get_error(
                    df[df["strategy"] == strategy], "rec_r", timesteps, ranges
                )
                error = np.abs(error)
                error_plot = format_error(error, est_timesteps)
                ax2.plot(list(range(350)), error_plot * 1000, **ERROR_KW)

                ax = ax2.twinx()
            else:
                ax = fig.add_subplot(sgs[0])

            [ax.axvspan(l[0] - 1, l[-1] + 1, zorder=5, **NO_DATA_KW) for l in no_data]
            sns.lineplot(data=df_gps, x=df_gps.index, y="Range [km]", ax=ax, **GPS_KW)
            sns.scatterplot(data=df[df["strategy"] == strategy], ax=ax, **RANGE_KW)
            ax.set_xlim(XLIM)
            ax.set_xticks(XTICKS)
            ax.set_xticklabels([])
            ax.set_xlabel(None)
            ax.set_ylim(YLIM_R)
            ax.set_title(strategy)

            # Switch twin y-axes
            if k > 0:
                primary_ticks = len(ax.yaxis.get_major_ticks())
                ax2.yaxis.set_major_locator(ticker.LinearLocator(primary_ticks))

                ax2.yaxis.tick_right()
                ax2.yaxis.set_label_position("right")

                ax.yaxis.tick_left()
                ax.yaxis.set_label_position("left")

                ax.grid(True)
                ax2.set_xlabel(None)
                ax2.set_yscale("log")
                ax2.set_ylim(YLIM_R_ERR)
                ax2.set_yticks(YTICKS_R_ERR)
                ax2.tick_params(
                    axis="y", which="both", length=0, colors=ERROR_KW["color"]
                )
                adjust_subplotticklabels(ax2, low=0)

            ax.set_yticks(YTICKS_R)
            ax.tick_params(axis="both", which="both", length=0)
            adjust_subplotticklabels(ax, 0, -1)

            if k == 6:
                ax.set_ylabel("Range [km]")
                if k > 0:
                    ax2.set_ylabel(
                        "Error [m]",
                        color=ERROR_KW["color"],
                        rotation=0,
                        va="bottom",
                        ha="right",
                        # y=1.08,
                        labelpad=0,
                    )
                    ax2.yaxis.set_label_coords(1.05, 1.08)
            else:
                ax.set_ylabel(None)
                if k > 0:
                    ax2.set_ylabel(None)

            # Depth ====================================================

            # Depth Error
            if k > 0:
                ax2 = fig.add_subplot(sgs[1])
                # ax2.grid(False)

                error, est_timesteps = get_error(
                    df[df["strategy"] == strategy], "src_z", timesteps, depths
                )
                error = np.abs(error)
                error_plot = format_error(error, est_timesteps)
                ax2.plot(list(range(350)), error_plot, **ERROR_KW)

                ax = ax2.twinx()
            else:
                ax = fig.add_subplot(sgs[1])

            [ax.axvspan(l[0] - 1, l[-1] + 1, zorder=1, **NO_DATA_KW) for l in no_data]
            sns.scatterplot(data=df[df["strategy"] == strategy], ax=ax, **DEPTH_KW)
            ax.set_xlabel(None)
            ax.set_xlim(XLIM)
            ax.set_ylim(ylim_z)

            # Switch twin y-axes
            if k > 0:
                primary_ticks = len(ax.yaxis.get_major_ticks())
                ax2.yaxis.set_major_locator(ticker.LinearLocator(primary_ticks))

                ax2.yaxis.tick_right()
                ax2.yaxis.set_label_position("right")

                ax.yaxis.tick_left()
                ax.yaxis.set_label_position("left")

                # ax.grid(True)
                ax2.set_yscale("log")
                ax2.set_ylim(YLIM_Z_ERR)
                ax2.set_yticks(YTICKS_Z_ERR)
                ax2.tick_params(axis="x", which="both", length=0)
                ax2.tick_params(
                    axis="y", which="both", length=0, colors=ERROR_KW["color"]
                )
                adjust_subplotticklabels(ax2, low=0)

            ax.set_xticks(XTICKS)
            ax.set_yticks(yticks_z)
            ax.tick_params(axis="both", which="both", length=0)
            adjust_subplotticklabels(ax, -1, 0)

            if k == 6:
                ax2.set_xlabel("Time Step")
                ax.set_ylabel("Depth [m]")
                if k > 0:
                    pass
                    # ax2.set_ylabel(
                    #     "Error [m]",
                    #     color=ERROR_KW["color"],
                    #     rotation=0,
                    #     va="top",
                    #     # ha="right",
                    #     x=-0.1,
                    #     y=-0.05,
                    #     labelpad=0,
                    # )
            else:
                ax.set_ylabel(None)
                if k > 0:
                    ax2.set_xlabel(None)
                    ax2.set_ylabel(None)

            # if k == 1:
            #     return fig
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
    loadpath = ROOT / "Data" / "range_estimation" / "demo"
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
            ax.text(0.1, 0.9, f"Iteration {trial}", va="top")

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


def simulations_localization():
    sim_dir = ROOT / "Data" / "localization" / "simulation"
    serial = "serial_230218"
    df = pd.read_csv(sim_dir / serial / "results" / "collated.csv")

    RANGES = [1.0, 3.0, 5.0, 7.0]
    TITLE_KW = {"ha": "left", "va": "top", "x": 0}

    fig, axs = plt.subplots(
        figsize=(16, 6), nrows=4, ncols=4, gridspec_kw={"wspace": 0.15}
    )

    # Column 1 - Objective Function ============================================
    TITLE = "Ambiguity surface: $f(\mathbf{x})$"
    XLIM = [0, 10]
    XLABEL = "Range [km]"
    YLIM = [0, 1.2]
    YLABEL = "Depth [km]"

    axcol = axs[:, 0]
    # Set ranges
    [
        axcol[i].text(
            -0.6,
            0.75,
            f"$R_\mathrm{{src}} = {r}$ km",
            transform=axcol[i].transAxes,
            fontsize="x-large",
        )
        for i, r in enumerate(RANGES)
    ]

    for ax, r in zip(axcol, RANGES):
        # TODO: Change from 1-D to 2-D ambiguity surface
        # fname = (
        #     ROOT
        #     / "Data"
        #     / "range_estimation"
        #     / "simulation"
        #     / "serial_230217"
        #     / f"rec_r={r:.1f}__src_z=60.0__snr=20"
        #     / "grid"
        #     / "seed_0002406475"
        #     / "results.json"
        # )

        selection = (
            (df["seed"] == int("0002406475"))
            & (df["strategy"] == "Grid")
            & (df["range"] == r)
        )
        df_obj = df[selection].sort_values("rec_r")
        g = sns.lineplot(data=df_obj, x="rec_r", y="bartlett", ax=ax)
        g.set(xlabel=None, ylabel=None)

    # Set x axis
    [axcol[i].set_xlim(XLIM) for i in range(len(RANGES))]
    [axcol[i].set_xticklabels([]) for i in range(len(RANGES) - 1)]
    # [axcol[-1].set_xlabel(None) for i in range(len(RANGES) - 1)]
    [axcol[-1].set_xlabel(XLABEL)]

    # Set y axis
    [axcol[i].set_ylim(YLIM) for i, _ in enumerate(RANGES)]
    axcol[-1].set_ylabel(YLABEL)

    # Set title
    axcol[0].set_title(TITLE, **TITLE_KW)

    # TODO: Indicate maximum with star

    # Column 2 - Performance History ===========================================
    TITLE = "Best observed: $\hat{f}(\mathbf{x})$"
    XLIM = [0, 801]
    XLABEL = "Evaluation"
    YLIM = [0, 1.2]

    axcol = axs[:, 1]

    for ax, r in zip(axcol, RANGES):
        selection = df["range"] == r
        g = sns.lineplot(
            data=df[selection],
            x="trial_index",
            y="best_values",
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
    TITLE = "Range error history: $\\vert\hat{R}_{src} - R_{src}\\vert$ [km]"
    YLIM = [0, 8]  # TODO: Validate this limit

    axcol = axs[:, 2]

    count = 0
    for ax, r in zip(axcol, RANGES):
        selection = df["range"] == r
        g = sns.lineplot(
            data=df[selection],
            x="trial_index",
            y="best_range_error",
            hue="strategy",
            ax=ax,
            legend="auto" if count == 3 else None,
        )
        g.set(xlabel=None, ylabel=None)
        if count == 3:
            sns.move_legend(
                ax, "upper center", bbox_to_anchor=(0.5, -0.5), ncol=4, title="Strategy"
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
    TITLE = "Depth error history: $\\vert\hat{z}_{src} - z_{src}\\vert$ [m]"
    YLIM = [200, 0]  # TODO: Validate this limit

    axcol = axs[:, 3]

    for ax, r in zip(axcol, RANGES):
        selection = df["range"] == r
        g = sns.lineplot(
            data=df[selection],
            x="trial_index",
            y="best_depth_error",
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


def simulations_range_est():
    SMOKE_TEST = False

    sim_dir = ROOT / "Data" / "range_estimation" / "simulation"
    serial = "serial_230217"
    if not SMOKE_TEST:
        df = pd.read_csv(sim_dir / serial / "results" / "collated.csv")

    RANGES = [1.0, 3.0, 5.0, 7.0]
    TITLE_KW = {"ha": "left", "va": "top", "x": 0}

    fig, axs = plt.subplots(
        figsize=(12, 6), nrows=4, ncols=3, gridspec_kw={"wspace": 0.15}
    )

    # Column 1 - Objective Function ============================================
    # TODO: Read in high-res objective function
    TITLE = "Ambiguity surface: $f(\mathbf{x})$"
    XLIM = [0, 10]
    XLABEL = "Range [km]"
    YLIM = [0, 1.2]

    axcol = axs[:, 0]
    # Set ranges
    [
        axcol[i].text(
            -0.52,
            0.5,
            f"$R_\mathrm{{src}} = {r}$ km",
            transform=axcol[i].transAxes,
            fontsize="large",
            ha="left",
        )
        for i, r in enumerate(RANGES)
    ]

    for ax, r in zip(axcol, RANGES):
        # fname = (
        #     ROOT
        #     / "Data"
        #     / "range_estimation"
        #     / "simulation"
        #     / "serial_230217"
        #     / f"rec_r={r:.1f}__src_z=60.0__snr=20"
        #     / "grid"
        #     / "seed_0002406475"
        #     / "results.json"
        # )

        if not SMOKE_TEST:
            selection = (
                (df["seed"] == int("0002406475"))
                & (df["strategy"] == "Grid")
                & (df["range"] == r)
            )
            df_obj = df[selection].sort_values("rec_r")
            g = sns.lineplot(data=df_obj, x="rec_r", y="bartlett", ax=ax)
            g.set(xlabel=None, ylabel=None)
            ax.axvline(r, color="r")

    # Set x axis
    [axcol[i].set_xlim(XLIM) for i in range(len(RANGES))]
    [axcol[i].set_xticklabels([]) for i in range(len(RANGES) - 1)]
    # [axcol[-1].set_xlabel(None) for i in range(len(RANGES) - 1)]
    [axcol[-1].set_xlabel(XLABEL)]

    # Set y axis
    [axcol[i].set_ylim(YLIM) for i, _ in enumerate(RANGES)]

    # Set title
    axcol[0].set_title(TITLE, **TITLE_KW)

    # Column 2 - Performance History ===========================================
    TITLE = "Best observed: $\hat{f}(\mathbf{x})$"
    XLIM = [0, 201]
    XLABEL = "Evaluation"
    YLIM = [0, 1.2]

    axcol = axs[:, 1]

    count = 0
    for ax, r in zip(axcol, RANGES):
        if not SMOKE_TEST:
            selection = df["range"] == r
            g = sns.lineplot(
                data=df[selection],
                x="trial_index",
                y="best_values",
                hue="strategy",
                ax=ax,
                legend="auto" if count == 3 else None,
            )
            g.set(xlabel=None, ylabel=None)
            if count == 3:
                sns.move_legend(
                    ax,
                    "upper center",
                    bbox_to_anchor=(0.5, -0.5),
                    ncol=4,
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

    # Column 3 - Error History =================================================
    TITLE = "Error history: $\\vert\hat{R}_{src} - R_{src}\\vert$ [km]"
    YLIM = [0, 8]

    axcol = axs[:, 2]

    for ax, r in zip(axcol, RANGES):
        if not SMOKE_TEST:
            selection = df["range"] == r
            g = sns.lineplot(
                data=df[selection],
                x="trial_index",
                y="best_range_error",
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
    figures = list(map(lambda i: int(i.strip()), args.figures.split(",")))
    main(figures)
