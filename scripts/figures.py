#!/usr/bin/env python3
"""Usage:
python3 ./Source/scripts/figures.py
python3 ./Source/scripts/figures.py 1,2,3
"""
import argparse
from pathlib import Path
import pickle
import sys
import warnings

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path.cwd() / "Source"))
from BOGP import utils
from BOGP.optimization import plotting
from BOGP.optimization.optimizer import import_from_str, Results
from tritonoa.kraken import run_kraken
from tritonoa.io import read_ssp
from tritonoa.sp import beamformer, snrdb_to_sigma, added_wng

ROOT = Path.home() / "Research" / "Projects" / "BOGP"
FIGURE_PATH = ROOT / "Reports" / "JASA" / "figures"

DPI = 200


def main(figures: list):
    for figure in figures:
        print(f"Producing Figure {figure:02d} " + 60 * "-")
        try:
            eval(f"figure{figure}()")
        except NameError:
            warnings.warn(f"Figure {figure} is not defined or implemented yet.")
            continue


def figure9():
    simulations_dashboard()


def figure1():
    # Set the true parameters:
    fig = plt.figure(figsize=(6, 10))
    gs = GridSpec(nrows=4, ncols=1, figure=fig, height_ratios=[1, 1, 1, 1])
    true_ranges = [1.0, 3.0, 5.0, 7.0]

    for j, range_true in enumerate(true_ranges):
        depth_true = 62
        freq = 201

        # Load CTD data
        z_data, c_data, _ = read_ssp(
            ROOT / "Data" / "SWELLEX96" / "CTD" / "i9606.prn", 0, 3, header=None
        )
        z_data = np.append(z_data, 217)
        c_data = np.append(c_data, c_data[-1])

        fixed_parameters = {
            # General
            "title": "SWELLEX96_SIM",
            "tmpdir": ".",
            "model": "KRAKENC",
            # Top medium
            # Layered media
            "layerdata": [
                {"z": z_data, "c_p": c_data, "rho": 1},
                {"z": [217, 240], "c_p": [1572.37, 1593.02], "rho": 1.8, "a_p": 0.3},
                {"z": [240, 1040], "c_p": [1881, 3245.8], "rho": 2.1, "a_p": 0.09},
            ],
            # Bottom medium
            "bot_opt": "A",
            "bot_c_p": 5200,
            "bot_rho": 2.7,
            "bot_a_p": 0.03,
            # Speed constraints
            "clow": 0,
            "chigh": 1600,
            # Receiver parameters
            "rec_z": np.linspace(94.125, 212.25, 64),
            "rec_r": range_true,
            "snr": 20,
            # Source parameters
            "src_z": depth_true,
            "freq": freq,
        }

        # Run simulation with true parameters:
        sigma = snrdb_to_sigma(fixed_parameters["snr"])
        p_rec = run_kraken(fixed_parameters)
        p_rec /= np.linalg.norm(p_rec)
        noise = added_wng(p_rec.shape, sigma=sigma, cmplx=True)
        p_rec += noise
        K = p_rec.dot(p_rec.conj().T)

        # Define parameter search space:
        [fixed_parameters.pop(item) for item in ["rec_r", "src_z"]]
        search_parameters = [
            {"name": "rec_r", "bounds": [0.001, 10.0]},
            {"name": "src_z", "bounds": [0.5, 200.0]},
        ]

        # Define search grid & run MFP
        dr = 5 / 1e3  # [km]
        dz = 2  # [m]
        rvec = np.arange(
            search_parameters[0]["bounds"][0],
            search_parameters[0]["bounds"][1] + dr,
            dr,
        )
        zvec = np.arange(
            search_parameters[1]["bounds"][0], search_parameters[1]["bounds"][1], dz
        )

        pbar = tqdm(
            zvec,
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
            desc="  MFP",
            leave=True,
            position=0,
            unit=" step",
        )
        p_rep = np.zeros((len(zvec), len(rvec), len(fixed_parameters["rec_z"])))
        B_bart = np.zeros((len(zvec), len(rvec)))

        for zz, z in enumerate(pbar):
            p_rep = run_kraken(fixed_parameters | {"src_z": z, "rec_r": rvec})
            for rr, r in enumerate(rvec):
                B_bart[zz, rr] = beamformer(K, p_rep[:, rr], atype="cbf").item()

        utils.clean_up_kraken_files(".")

        # Bn = B_bart / np.max(B_bart)
        Bn = B_bart
        logBn = 10 * np.log10(Bn)
        src_z_ind, src_r_ind = np.unravel_index(
            np.argmax(logBn), (len(zvec), len(rvec))
        )

        ax = fig.add_subplot(gs[j])
        im = ax.imshow(
            logBn,
            aspect="auto",
            extent=[min(rvec), max(rvec), min(zvec), max(zvec)],
            origin="lower",
            vmin=-10,
            vmax=0,
            interpolation="none",
            cmap="jet",
        )
        ax.plot(
            rvec[src_r_ind],
            zvec[src_z_ind],
            "w*",
            markersize=15,
            markeredgewidth=1.5,
            markeredgecolor="k",
        )
        ax.invert_yaxis()
        ax.set_xlim([0, 10])
        ax.set_ylim([200, 0])
        if j != 3:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        if j == 3:
            ax.set_xlabel("Range [km]")
            ax.set_ylabel("Depth [m]")
            # ax_div = make_axes_locatable(ax)

            # im.colorbar(label="Normalized Correlation [dB]", ax=ax)
        ax.set_title(f"R = {range_true:.1f} km, z = 62 m")

    cax = inset_axes(
        ax,
        width="100%",
        height="10%",
        loc="center",
        bbox_to_anchor=(0, -0.85, 1, 1),
        bbox_transform=ax.transAxes,
    )
    fig.colorbar(
        im, cax=cax, label="Normalized Correlation [dB]", orientation="horizontal"
    )
    fig.savefig(
        FIGURE_PATH / "MFP.png",
        dpi=DPI,
        facecolor="white",
        bbox_inches="tight",
    )


def figure2():
    DATAPATH = ROOT / "Data" / "Simulations"
    EXPERIMENT = DATAPATH / "Protected" / "range_estimation"
    DATA_FOLDER = (
        EXPERIMENT
        / "acq_func=ExpectedImprovement__snr=Inf__rec_r=3.0"
        / "Runs"
        / "0063373286"
    )
    # eval_index = np.linspace(0, 80, 9, dtype=int).tolist()
    eval_index = [0, 1, 2, 10, 20, 30, 50, 70, 80]
    optim_config = torch.load(
        DATA_FOLDER / "optim.pth", map_location=torch.device("cpu")
    )
    results = Results().load(DATA_FOLDER / "results.pth")

    model_data = np.load(DATA_FOLDER.parent.parent / "measured" / "measured.npz")
    K = model_data["covariance"]
    p_rec = model_data["pressure"]
    with open(DATA_FOLDER.parent.parent / "measured" / "parameters.pkl", "rb") as f:
        parameters = pickle.load(f)

    optim_config["obj_func_kwargs"] = {"K": K, "parameters": parameters}

    search_parameters = [{"name": "x1", "bounds": [0.001, 10]}]
    X_test = torch.linspace(
        search_parameters[0]["bounds"][0], search_parameters[0]["bounds"][1], 1001
    ).unsqueeze(1)
    pltr = plotting.ResultsPlotter(optim_config, results)
    parameters_to_plot = ["x1"]
    fig = pltr.plot_training_iterations(
        X_test,
        parameters_to_plot,
        index=eval_index,
        parameter_labels="$\mathbf{x}$ (Range [km])",
        consolidate=True,
    )
    fig.savefig(
        FIGURE_PATH / "optim_1D.png", dpi=DPI, facecolor="white", bbox_inches="tight"
    )


def figure3():
    zw, cw, _ = read_ssp(
        ROOT / "Data" / "SWELLEX96" / "CTD" / "i9606.prn", 0, 3, header=None
    )
    zw = np.append(zw, 217)
    cw = np.append(cw, cw[-1])
    zb1 = np.array([np.NaN, 217, 240])
    cb1 = np.array([np.NaN, 1572.37, 1593.02])
    zb2 = np.array([np.NaN, 240, 1040])
    cb2 = np.array([np.NaN, 1881, 3245.8])
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

    fig.savefig(
        FIGURE_PATH / "environment.png", dpi=DPI, facecolor="white", bbox_inches="tight"
    )


def figure4():
    evaluations = {
        "acq_func": [
            "ProbabilityOfImprovement",
            "ExpectedImprovement",
            "qExpectedImprovement",
            "random",
        ],
        "acq_func_abbrev": ["PI", "EI", "qEI", "Rand"],
        "snr": ["inf", "10"],
        "rec_r": ["0.5", "3.0", "6.0", "10.0"],
    }
    EXPERIMENT1 = ROOT / "Data" / "Simulations" / "bogp" / "range_estimation"
    df1 = pd.read_csv(EXPERIMENT1 / "aggregated.csv", index_col=0)
    df1["best_param"] = df1["best_param"].str.strip("[").str.strip("]").astype(float)

    EXPERIMENT2 = ROOT / "Data" / "Simulations" / "random" / "range_estimation"
    df2 = pd.read_csv(EXPERIMENT2 / "aggregated.csv", index_col=0)
    df2["best_param"] = df2["best_param"].str.strip("[").str.strip("]").astype(float)

    df = pd.concat([df1, df2])

    fig = plotting.plot_aggregated_data(
        df,
        evaluations,
        "best_value",
        optimum=1.0,
        upper_threshold=1.0,
        title="Performance History",
        xlabel="Evaluation",
        ylabel="$\hat{f}(\mathbf{x})$",
        xlim=[-2, 202],
        ylim=[0, 1.05],
    )
    fig.savefig(
        FIGURE_PATH / "RangeEst_PerfHist.png",
        dpi=DPI,
        facecolor="white",
        bbox_inches="tight",
    )


def figure5():
    evaluations = {
        "acq_func": [
            "ProbabilityOfImprovement",
            "ExpectedImprovement",
            "qExpectedImprovement",
            "random",
        ],
        "acq_func_abbrev": ["PI", "EI", "qEI", "Rand"],
        "snr": ["inf", "10"],
        "rec_r": ["0.5", "3.0", "6.0", "10.0"],
    }
    EXPERIMENT1 = ROOT / "Data" / "Simulations" / "bogp" / "range_estimation"
    df1 = pd.read_csv(EXPERIMENT1 / "aggregated.csv", index_col=0)
    df1["best_param"] = df1["best_param"].str.strip("[").str.strip("]").astype(float)

    EXPERIMENT2 = ROOT / "Data" / "Simulations" / "random" / "range_estimation"
    df2 = pd.read_csv(EXPERIMENT2 / "aggregated.csv", index_col=0)
    df2["best_param"] = df2["best_param"].str.strip("[").str.strip("]").astype(float)

    df = pd.concat([df1, df2])

    fig = plotting.plot_aggregated_data(
        df,
        evaluations,
        "best_param",
        compute_error_with=evaluations["rec_r"],
        title="Error History",
        xlabel="Evaluation",
        ylabel="$\\vert\hat{r}_{src} - r_{src}\\vert$",
        xlim=[-2, 202],
        ylim=[0, 10],
    )
    fig.savefig(
        FIGURE_PATH / "RangeEst_ErrHist.png",
        dpi=DPI,
        facecolor="white",
        bbox_inches="tight",
    )


def figure6():
    evaluations = {
        "acq_func": [
            "ProbabilityOfImprovement",
            "ExpectedImprovement",
            "qExpectedImprovement",
            "random",
        ],
        "acq_func_abbrev": ["PI", "EI", "qEI", "Rand"],
        "snr": ["inf", "10"],
        "rec_r": ["0.5", "3.0", "6.0", "10.0"],
        "src_z": ["62"],
    }
    EXPERIMENT1 = ROOT / "Data" / "Simulations" / "bogp" / "localization"
    df1 = pd.read_csv(EXPERIMENT1 / "aggregated.csv", index_col=0)
    new_cols = (
        df1["best_param"]
        .str.strip("[ ")
        .str.strip(" ]")
        .str.split(" ", n=1, expand=True)
    )
    new_cols.columns = [f"best_param{col}" for col in new_cols.columns]
    df1 = pd.concat([df1, new_cols], axis=1).drop("best_param", axis=1)

    EXPERIMENT2 = ROOT / "Data" / "Simulations" / "random" / "localization"
    df2 = pd.read_csv(EXPERIMENT2 / "aggregated.csv", index_col=0)
    new_cols = (
        df2["best_param"]
        .str.strip("[ ")
        .str.strip(" ]")
        .str.split(" ", n=1, expand=True)
    )
    new_cols.columns = [f"best_param{col}" for col in new_cols.columns]
    df2 = pd.concat([df2, new_cols], axis=1).drop("best_param", axis=1)

    df = pd.concat([df1, df2])

    fig = plotting.plot_aggregated_data(
        df,
        evaluations,
        "best_value",
        optimum=1.0,
        upper_threshold=1.0,
        title="Performance History",
        xlabel="Evaluation",
        ylabel="$\hat{f}(\mathbf{x})$",
        xlim=[-5, 1005],
        ylim=[0, 1.05],
    )
    fig.savefig(
        FIGURE_PATH / "Localization_PerfHist.png",
        dpi=DPI,
        facecolor="white",
        bbox_inches="tight",
    )


def figure7():
    evaluations = {
        "acq_func": [
            "ProbabilityOfImprovement",
            "ExpectedImprovement",
            "qExpectedImprovement",
            "random",
        ],
        "acq_func_abbrev": ["PI", "EI", "qEI", "Rand"],
        "snr": ["inf", "10"],
        "rec_r": ["0.5", "3.0", "6.0", "10.0"],
        "src_z": ["62"],
    }
    EXPERIMENT1 = ROOT / "Data" / "Simulations" / "bogp" / "localization"
    df1 = pd.read_csv(EXPERIMENT1 / "aggregated.csv", index_col=0)
    new_cols = (
        df1["best_param"]
        .str.strip("[ ")
        .str.strip(" ]")
        .str.split(" ", n=1, expand=True)
    )
    new_cols.columns = [f"best_param{col}" for col in new_cols.columns]
    df1 = pd.concat([df1, new_cols], axis=1).drop("best_param", axis=1)

    EXPERIMENT2 = ROOT / "Data" / "Simulations" / "random" / "localization"
    df2 = pd.read_csv(EXPERIMENT2 / "aggregated.csv", index_col=0)
    new_cols = (
        df2["best_param"]
        .str.strip("[ ")
        .str.strip(" ]")
        .str.split(" ", n=1, expand=True)
    )
    new_cols.columns = [f"best_param{col}" for col in new_cols.columns]
    df2 = pd.concat([df2, new_cols], axis=1).drop("best_param", axis=1)

    df = pd.concat([df1, df2])

    fig = plotting.plot_aggregated_data(
        df,
        evaluations,
        "best_param0",
        compute_error_with=evaluations["rec_r"],
        title="Range Error History",
        xlabel="Evaluation",
        ylabel="$\\vert\hat{r}_{src} - r_{src}\\vert$",
        xlim=[-5, 1005],
        ylim=[0, 10],
    )
    fig.savefig(
        FIGURE_PATH / "Localization_ErrHistRange.png",
        dpi=DPI,
        facecolor="white",
        bbox_inches="tight",
    )


def figure8():
    evaluations = {
        "acq_func": [
            "ProbabilityOfImprovement",
            "ExpectedImprovement",
            "qExpectedImprovement",
            "random",
        ],
        "acq_func_abbrev": ["PI", "EI", "qEI", "Rand"],
        "snr": ["inf", "10"],
        "rec_r": ["0.5", "3.0", "6.0", "10.0"],
        "src_z": ["62"],
    }
    EXPERIMENT1 = ROOT / "Data" / "Simulations" / "bogp" / "localization"
    df1 = pd.read_csv(EXPERIMENT1 / "aggregated.csv", index_col=0)
    new_cols = (
        df1["best_param"]
        .str.strip("[ ")
        .str.strip(" ]")
        .str.split(" ", n=1, expand=True)
    )
    new_cols.columns = [f"best_param{col}" for col in new_cols.columns]
    df1 = pd.concat([df1, new_cols], axis=1).drop("best_param", axis=1)

    EXPERIMENT2 = ROOT / "Data" / "Simulations" / "random" / "localization"
    df2 = pd.read_csv(EXPERIMENT2 / "aggregated.csv", index_col=0)
    new_cols = (
        df2["best_param"]
        .str.strip("[ ")
        .str.strip(" ]")
        .str.split(" ", n=1, expand=True)
    )
    new_cols.columns = [f"best_param{col}" for col in new_cols.columns]
    df2 = pd.concat([df2, new_cols], axis=1).drop("best_param", axis=1)

    df = pd.concat([df1, df2])

    fig = plotting.plot_aggregated_data(
        df,
        evaluations,
        "best_param1",
        compute_error_with=float(evaluations["src_z"][0]),
        title="Depth Error History",
        xlabel="Evaluation",
        ylabel="$\\vert\hat{z}_{src} - z_{src}\\vert$",
        xlim=[-5, 1005],
        ylim=[0, 100],
    )
    fig.savefig(
        FIGURE_PATH / "Localization_ErrHistDepth.png",
        dpi=DPI,
        facecolor="white",
        bbox_inches="tight",
    )


def simulations_dashboard():
    # folders = utils.folders_of_evaluations(evaluations)
    EXPERIMENT1 = ROOT / "Data" / "Simulations" / "bogp" / "range_estimation"
    df1 = pd.read_csv(EXPERIMENT1 / "aggregated.csv", index_col=0)
    df1["best_param"] = df1["best_param"].str.strip("[").str.strip("]").astype(float)
    df1 = df1[df1["snr"] == 10]

    EXPERIMENT2 = ROOT / "Data" / "Simulations" / "random" / "range_estimation"
    df2 = pd.read_csv(EXPERIMENT2 / "aggregated.csv", index_col=0)
    df2["best_param"] = df2["best_param"].str.strip("[").str.strip("]").astype(float)
    df2 = df2[df2["snr"] == 10]

    dfr = pd.concat([df1, df2])
    del df1, df2

    EXPERIMENT1 = ROOT / "Data" / "Simulations" / "bogp" / "localization"
    df1 = pd.read_csv(EXPERIMENT1 / "aggregated.csv", index_col=0)
    new_cols = df1["best_param"].str.strip("[ ").str.strip(" ]").str.split(" ", n=1, expand=True)
    new_cols.columns = [f"best_param{col}" for col in new_cols.columns]
    df1 = pd.concat([df1, new_cols], axis=1).drop("best_param", axis=1)
    df1 = df1[df1["snr"] == 10]

    EXPERIMENT2 = ROOT / "Data" / "Simulations" / "random" / "localization"
    df2 = pd.read_csv(EXPERIMENT2 / "aggregated.csv", index_col=0)
    new_cols = df2["best_param"].str.strip("[ ").str.strip(" ]").str.split(" ", n=1, expand=True)
    new_cols.columns = [f"best_param{col}" for col in new_cols.columns]
    df2 = pd.concat([df2, new_cols], axis=1).drop("best_param", axis=1)
    df2 = df2[df2["snr"] == 10]

    dfl = pd.concat([df1, df2])
    del df1, df2

    fig, axs = plt.subplots(
        nrows=4,
        ncols=6,
        figsize=(18, 6),
        facecolor="white",
        dpi=200,
        gridspec_kw={"width_ratios": [2, 1, 1, 1, 1, 1], "wspace": 0.4},
    )
    # ranges = df["rec_r"].unique()

    # Subplot letters
    letters = "abcdef"
    [
        axs[0, i].text(
            -0.25,
            1.1,
            letters[i] + ")",
            transform=axs[0, i].transAxes,
            fontsize="xx-large",
        )
        for i in range(len(letters))
    ]

    simulations = {
        "acq_func": [
            "ProbabilityOfImprovement",
            "ExpectedImprovement",
            "qExpectedImprovement",
            "random",
        ],
        "acq_func_abbrev": ["PI", "EI", "qEI", "Rand"],
        "snr": ["inf", "10"],
        "rec_r": ["0.5", "3.0", "6.0", "10.0"],
        "src_z": ["62"],
    }
    ranges = [float(i) for i in simulations["rec_r"]]

    # ================================================================ #
    # ======================= Ambiguity Surfaces ===================== #
    # ================================================================ #

    axcol = axs[:, 0]

    XLIM = [0, 10]
    YLIM = [200, 0]
    XLABEL = "Range [km]"
    YLABEL = "Depth [m]"

    # Draw ranges
    [
        axcol[i].text(
            -0.8,
            0.5,
            f"$r_\mathrm{{src}} = {r}$ km",
            transform=axcol[i].transAxes,
            fontsize="xx-large",
        )
        for i, r in enumerate(ranges)
    ]

    # Set x axis
    [axcol[i].set_xlim(XLIM) for i in range(len(ranges))]
    [axcol[i].set_xticklabels([]) for i in range(len(ranges) - 1)]
    [axcol[-1].set_xlabel(XLABEL)]

    # Set y axis
    [axcol[i].set_ylim(YLIM) for i, _ in enumerate(ranges)]
    axcol[-1].set_ylabel(YLABEL)

    for i, r in enumerate(ranges):
        data = np.load(
            ROOT / "Data" / "SWELLEX96" / "ambiguity_surfaces" / f"201Hz_62m_{r}km.npz"
        )
        B = data["B"]
        rvec = data["rvec"]
        zvec = data["zvec"]
        ax, im = plotting.plot_ambiguity_surface(B, rvec, zvec, axcol[i])

    cax = inset_axes(
        ax,
        width="100%",
        height="10%",
        loc="center",
        bbox_to_anchor=(0, -1.1, 1, 1),
        bbox_transform=ax.transAxes,
    )
    fig.colorbar(
        im, cax=cax, label="Normalized Correlation [dB]", orientation="horizontal"
    )
    # ================================================================ #
    # ======================== Range Estimation ====================== #
    # ================================================================ #

    XLABEL = "Evaluations"
    # ======================= Performance History ==================== #
    axcol = axs[:, 1]
    VALUE_TO_PLOT = "best_value"
    XLIM = [-5, 205]
    YLIM = [0, 1.05]
    YLABEL = "$\hat{f}(\mathbf{x})$"
    OPTIMUM = 1.0
    LOWER_THRESHOLD = 0
    UPPER_THRESHOLD = 1

    # Set x axis
    [axcol[i].set_xlim(XLIM) for i in range(len(ranges))]
    [axcol[i].set_xticklabels([]) for i in range(len(ranges) - 1)]
    [axcol[-1].set_xlabel(XLABEL)]

    # Set y axis
    [axcol[i].set_ylim(YLIM) for i in range(len(ranges))]
    axcol[-1].set_ylabel(YLABEL)

    # Draw plots
    plots = [
        plotting.plot_agg_data(
            dfr[dfr["rec_r"] == r],
            simulations,
            VALUE_TO_PLOT,
            optimum=OPTIMUM,
            lower_threshold=LOWER_THRESHOLD,
            upper_threshold=UPPER_THRESHOLD,
            ax=axcol[i],
        )
        for i, r in enumerate(ranges)
    ]

    # ====================== Range Error History ===================== #
    axcol = axs[:, 2]
    VALUE_TO_PLOT = "best_param"
    YLIM = [0, 10]
    YLABEL = "$\\vert\hat{r}_{src} - r_{src}\\vert$"
    COMPUTE_ERROR_WITH = simulations["rec_r"]

    # Set x axis
    [axcol[i].set_xlim(XLIM) for i in range(len(ranges))]
    [axcol[i].set_xticklabels([]) for i in range(len(ranges) - 1)]
    [axcol[-1].set_xlabel(XLABEL)]

    # Set y axis
    [axcol[i].set_ylim(YLIM) for i in range(len(ranges))]
    axcol[-1].set_ylabel(YLABEL)

    # Draw plots
    plots = [
        plotting.plot_agg_data(
            dfr[dfr["rec_r"] == r],
            simulations,
            VALUE_TO_PLOT,
            compute_error_with=float(COMPUTE_ERROR_WITH[i]),
            ax=axcol[i],
        )
        for i, r in enumerate(ranges)
    ]

    # ================================================================ #
    # ========================== Localization ======================== #
    # ================================================================ #

    # ======================= Performance History ==================== #
    axcol = axs[:, 3]
    VALUE_TO_PLOT = "best_value"
    XLIM = [-5, 1005]
    YLIM = [0, 1.05]
    YLABEL = "$\hat{f}(\mathbf{x})$"
    OPTIMUM = 1.0
    LOWER_THRESHOLD = 0
    UPPER_THRESHOLD = 1

    # Set x axis
    [axcol[i].set_xlim(XLIM) for i in range(len(ranges))]
    [axcol[i].set_xticklabels([]) for i in range(len(ranges) - 1)]
    [axcol[-1].set_xlabel(XLABEL)]

    # Set y axis
    [axcol[i].set_ylim(YLIM) for i in range(len(ranges))]
    axcol[-1].set_ylabel(YLABEL)

    # Draw plots
    lines = [
        plotting.plot_agg_data(
            dfl[dfl["rec_r"] == r],
            simulations,
            VALUE_TO_PLOT,
            optimum=OPTIMUM,
            lower_threshold=LOWER_THRESHOLD,
            upper_threshold=UPPER_THRESHOLD,
            ax=axcol[i],
        )
        for i, r in enumerate(ranges)
    ]

    axcol[-1].legend(
        handles=lines[-1], ncol=5, loc="center", bbox_to_anchor=(0.5, -0.75)
    )

    # ====================== Range Error History ===================== #
    axcol = axs[:, 4]
    VALUE_TO_PLOT = "best_param0"
    YLIM = [0, 10]
    YLABEL = "$\\vert\hat{r}_{src} - r_{src}\\vert$"
    COMPUTE_ERROR_WITH = simulations["rec_r"]

    # Set x axis
    [axcol[i].set_xlim(XLIM) for i in range(len(ranges))]
    [axcol[i].set_xticklabels([]) for i in range(len(ranges) - 1)]
    [axcol[-1].set_xlabel(XLABEL)]

    # Set y axis
    [axcol[i].set_ylim(YLIM) for i in range(len(ranges))]
    axcol[-1].set_ylabel(YLABEL)

    # Draw plots
    lines = [
        plotting.plot_agg_data(
            dfl[dfl["rec_r"] == r],
            simulations,
            VALUE_TO_PLOT,
            compute_error_with=float(COMPUTE_ERROR_WITH[i]),
            ax=axcol[i],
        )
        for i, r in enumerate(ranges)
    ]

    # ====================== Depth Error History ===================== #
    axcol = axs[:, 5]
    VALUE_TO_PLOT = "best_param1"
    YLIM = [0, 100]
    YLABEL = "$\\vert\hat{z}_{src} - z_{src}\\vert$"
    COMPUTE_ERROR_WITH = [simulations["src_z"][0] for i in range(len(ranges))]

    # Set x axis
    [axcol[i].set_xlim(XLIM) for i in range(len(ranges))]
    [axcol[i].set_xticklabels([]) for i in range(len(ranges) - 1)]
    [axcol[-1].set_xlabel(XLABEL)]

    # Set y axis
    [axcol[i].set_ylim(YLIM) for i in range(len(ranges))]
    axcol[-1].set_ylabel(YLABEL)

    # Draw plots
    lines = [
        plotting.plot_agg_data(
            dfl[dfl["rec_r"] == r],
            simulations,
            VALUE_TO_PLOT,
            compute_error_with=float(COMPUTE_ERROR_WITH[i]),
            ax=axcol[i],
        )
        for i, r in enumerate(ranges)
    ]

    fig.savefig(
        FIGURE_PATH / "sim_dashboard.png",
        dpi=DPI,
        facecolor="white",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("figures", type=str)
    args = parser.parse_args()
    figures = list(map(lambda i: int(i.strip()), args.figures.split(",")))
    main(figures)
