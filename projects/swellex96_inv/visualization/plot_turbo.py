#!/usr/bin/env python

from pathlib import Path
from string import ascii_lowercase
import sys
from typing import Optional

from matplotlib import gridspec
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np
import pandas as pd
import seaborn as sns
import scienceplots

sys.path.insert(0, str(Path(__file__).parents[1]))
from data.bo import common, helpers

plt.style.use(["science", "ieee"])
# plt.style.use(["science", "ieee", "std-colors"])

SAVEFIG_KWARGS = {"dpi": 300, "bbox_inches": "tight"}
NBINS = 50
STRATEGY_COLORS = {
    "Random": "tab:red",
    "BO-LogEI": "tab:orange",
    "TuRBO (q=1)": "tab:cyan",
    "TuRBO (q=4)": "tab:green",
}
SORTING_RULE = {
    "Random": 0,
    "BO-LogEI": 1,
    "TuRBO (q=1)": 2,
    "TuRBO (q=4)": 3,
}


MODES = ["Simulated", "Experimental"]
N_INIT = 32
N_TRIALS = 100
N_RAND = 100
YLIM = [(-0.01, 0.4), (0.0, 0.4)]
YTICKS = [[0.0, 0.2, 0.4], [0.0, 0.2, 0.4]]
YTICKLABELS = YTICKS
DIST_LABELS = {
    "Random": "Random",
    "BO-LogEI": "BO-LogEI",
    "TuRBO (q=1)": "TuRBO ($q=1$)",
    "TuRBO (q=4)": "TuRBO ($q=4$)",
}


def print_hypervolume():
    dim = 7
    lengths = np.array([0.8, 0.4, 0.2, 0.1, 0.05])
    print(lengths)
    print(lengths**dim)


def get_err_matrix(df: pd.DataFrame, mode: str) -> None:
    df = df[df["Mode"] == mode]
    sel = df["Trial"] == N_TRIALS
    df = df.loc[sel]

    strategies = sorted(
        list(df["Strategy"].unique()), key=SORTING_RULE.__getitem__
    )
    df = df.drop(
        columns=[
            "seed",
            "wall_time",
            "Trial",
            "n_iter",
            "n_init",
            "obj",
            "rec_r",
            "src_z",
            "tilt",
            "h_w",
            "h_sed",
            "c_p_sed_top",
            "dc_p_sed",
            "c_p_sed_bot",
            "best_rec_r",
            "best_src_z",
            "best_tilt",
            "best_h_w",
            "best_h_sed",
            "best_c_p_sed_top",
            "best_dc_p_sed",
            "best_c_p_sed_bot",
            "best_dc_p_sed_err",
            "Mode",
        ]
    )
    df = df.groupby("Strategy").mean()
    df = df.reindex(strategies)
    print(df)
    print(df.to_latex(float_format="%.3f"))


def performance_plot(df: pd.DataFrame) -> plt.Figure:
    fig, axs = plt.subplots(
        nrows=1,
        ncols=len(MODES),
        figsize=(6.5, 1.5),
        gridspec_kw={"wspace": 0.08},
    )

    panel_configs = [
        [
            {"ylim": (1e-5, 0.3), "log": True, "plot_key": "best_obj"},
            {"ylim": (7e-2, 0.4), "log": True, "plot_key": "best_obj"},
        ],
        [
            {
                "ylim": (0, 50.0),
                "log": False,
                "plot_key": "cum_regret",
                "yticks": [0.1, 0.2, 0.3],
                "formatter": "%.1f",
            },
            {
                "ylim": (0, 50.0),
                "log": False,
                "plot_key": "cum_regret",
                "yticks": [0.1, 0.2, 0.3],
                "formatter": "%.1f",
            },
        ],
    ]

    legend_handles = None
    legend_labels = None

    for col, (mode, side) in enumerate(zip(MODES, ["left", "right"])):

        ax = axs[col]
        ax.set_title(f"{mode} Results")
        ax.text(0, 1.05, f"({ascii_lowercase[col]})", transform=ax.transAxes)
        config = panel_configs[0][col]
        ax = plot_panel(
            df[df["Mode"] == mode],
            ax,
            config["ylim"],
            side,
            plot_key=config["plot_key"],
            log=config["log"],
            yticks=config.get("yticks"),
            formatter=config.get("formatter"),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Get legend handles and labels from the first subplot's left panel
        if col == 0:
            left_ax = fig.axes[-2]  # The left panel of the current subplot
            legend_handles, legend_labels = left_ax.get_legend_handles_labels()

    # Add common legend at the bottom of the figure
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=len(legend_handles),
    )
    return fig


def plot_panel(
    df: pd.DataFrame,
    ax: plt.Axes,
    ylim: tuple,
    side: str,
    plot_key: str = "best_obj",
    log: bool = False,
    yticks: Optional[list] = None,
    formatter: Optional[str] = None,
    ylabel: str = "$\widehat{\phi}(t)$",
) -> plt.Axes:

    subplotspec = ax.get_subplotspec()
    gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplotspec, wspace=0.0, width_ratios=[9, 1]
    )

    ax_left = ax.figure.add_subplot(gs[0, 0])
    ax_left = plot_performance_history(df, ax=ax_left, plot_key=plot_key)
    ax_left.grid(axis="y", which="major")
    ax_left.axvline(x=N_INIT, color="k", linestyle="--", linewidth=0.75)
    ax_left.set_xlabel("Trial")
    ax_left.xaxis.set_label_coords(0.05, -0.04)
    if log:
        ax_left.set_yscale("log")
        ax_left.yaxis.set_major_locator(LogLocator(base=10, numticks=5))
    ax_left.set_xlim(30, 100)
    ax_left.set_ylim(ylim)
    ax_left.spines["right"].set_visible(False)

    ax_right = ax.figure.add_subplot(gs[0, 1])
    ax_right = plot_est_dist(df, ax=ax_right, plot_key=plot_key)
    ax_right.grid(axis="y", which="major")
    if log:
        ax_right.set_yscale("log")
    ax_right.set_ylim(ylim)
    ax_right.set_xlabel(None)
    ax_right.set_xticks([])
    ax_right.set_xticklabels([])
    ax_right.spines["left"].set_visible(False)
    ax_right.yaxis.tick_right()

    if side == "left":
        ax_left.set_ylabel(ylabel, rotation=0)
        ax_right.yaxis.set_major_formatter(plt.NullFormatter())
        ax_right.yaxis.set_minor_formatter(plt.NullFormatter())
        ax_right.set_ylabel(None)

        if formatter and not log:
            ax_left.yaxis.set_major_formatter(plt.FormatStrFormatter(formatter))
        if yticks:
            ax_left.set_yticks(yticks)

    if side == "right":
        ax_left.yaxis.set_major_formatter(plt.NullFormatter())
        ax_left.yaxis.set_minor_formatter(plt.NullFormatter())
        ax_left.set_ylabel(None)
        ax_right.set_ylabel(ylabel, rotation=0)
        ax_right.yaxis.set_label_position("right")

        if formatter and not log:
            ax_right.yaxis.set_major_formatter(plt.FormatStrFormatter(formatter))
        else:
            ax_right.yaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
        if yticks:
            ax_left.set_yticks(yticks)
            ax_right.set_yticks(yticks)
        else:
            ax_left.set_yticks([0.1, 0.2, 0.3, 0.4])
            ax_right.set_yticks([0.1, 0.2, 0.3, 0.4])

    return ax


def plot_performance_history(
    df: pd.DataFrame, ax: Optional[plt.Axes] = None, plot_key: str = "best_obj"
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()
    strategies = sorted(list(df["Strategy"].unique()), key=SORTING_RULE.__getitem__)

    for strategy in strategies:
        dfp = df.loc[df["Strategy"] == strategy].pivot(
            index="Trial", columns="seed", values=plot_key
        )

        mean = dfp.mean(axis=1)
        upper_quant = dfp.quantile(0.95, axis=1)
        lower_quant = dfp.quantile(0.05, axis=1)
        ax.plot(
            dfp.index,
            mean,
            linestyle="-",
            color=STRATEGY_COLORS[strategy],
            label=DIST_LABELS[strategy],
        )
        ax.plot(
            dfp.index,
            lower_quant,
            color=STRATEGY_COLORS[strategy],
            linestyle="-.",
            linewidth=0.5,
        )
        ax.plot(
            dfp.index,
            upper_quant,
            color=STRATEGY_COLORS[strategy],
            linestyle="-.",
            linewidth=0.5,
        )

    return ax


def plot_est_dist(
    df: pd.DataFrame, ax: Optional[plt.Axes] = None, plot_key: str = "best_obj"
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()

    sel = df["Trial"] == 100

    dfp = df.loc[sel].sort_values(
        by="Strategy", key=lambda x: x.apply(lambda y: SORTING_RULE.get(y, 1000))
    )
    sns.stripplot(
        data=dfp,
        x="Strategy",
        y=plot_key,
        hue="Strategy",
        palette=STRATEGY_COLORS,
        ax=ax,
        dodge=False,
        linewidth=0.5,
        alpha=0.5,
        size=3,
        marker="o",
    )
    return ax


def plot_param_dist(data: np.ndarray, bounds: list, ax: Optional[plt.Axes] = None):
    if ax is None:
        ax = plt.gca()

    sns.histplot(
        x=data, ax=ax, legend=None, element="step", bins=np.linspace(*bounds, NBINS)
    )
    ax.set_xlabel(None)
    ax.set_ylabel(None)
    return ax


def plot_parameter_estimates(
    data: list[pd.DataFrame], parameters: list[dict]
) -> plt.Figure:
    YLIM = [[0, 80], [0, 80]]

    fig, axs = plt.subplots(
        nrows=2, ncols=7, figsize=(6.5, 1.), gridspec_kw={"wspace": 0.1, "hspace": 0.4}
    )

    for i, axrow in enumerate(axs):
        df = data[i]
        for j, ax in enumerate(axrow):
            ax.grid(True, linestyle="dotted")

            param_name = common.SEARCH_SPACE[j]["name"]
            param_label = common.VARIABLES[common.SEARCH_SPACE[j]["name"]]
            bounds = common.SEARCH_SPACE[j]["bounds"]
            true_value = parameters[i][common.SEARCH_SPACE[j]["name"]]
            if param_name == "dc_p_sed":
                param_name = "c_p_sed_bot"
                param_label = common.VARIABLES[param_name]
                bounds = [1570, 1610]
                true_value = 1593.0

            ax = plot_param_dist(df[f"best_{param_name}"], bounds, ax=ax)

            ax.text(0, 1.07, f"({ascii_lowercase[7 * i + j]})", transform=ax.transAxes)
            ax.axvline(
                true_value,
                color="k",
                linestyle="dashed",
                linewidth=0.5,
            )
            ax.annotate(
                "",
                xy=(true_value, YLIM[i][1] - 3),
                xytext=(true_value, YLIM[i][1] + 10),
                arrowprops=dict(arrowstyle="simple", color="black"),
            )
            ax.set_xlim(bounds)
            ax.set_xticklabels([]) if i != 1 else None
            ax.set_xlabel(param_label) if i == 1 else None
            ax.set_ylim(YLIM[i])
            ax.set_yticklabels([]) if j != 0 else None
            ax.set_ylabel(f"Freq.\n({MODES[i][0:3]}.)") if j == 0 else None
            helpers.adjust_subplotxticklabels(ax, 0, -1)

    return fig


def main(n_init: int = 32) -> plt.Figure:
    print_hypervolume()
    return
    path = common.SWELLEX96Paths.outputs / "runs"

    if not (common.SWELLEX96Paths.outputs / "icassp26_results.csv").exists():
        df_rand_sim = helpers.load_data(
            path / "sim_random",
            "*.npz",
            common.SEARCH_SPACE,
            common.TRUE_SIM_VALUES,
            include_time=False,
        )
        df_rand_sim = df_rand_sim[df_rand_sim["Trial"] <= 100]
        df_rand_sim["Mode"] = "Simulated"

        df_rand_exp = helpers.load_data(
            path / "exp_random",
            "*.npz",
            common.SEARCH_SPACE,
            common.TRUE_EXP_VALUES,
            include_time=False,
        )
        df_rand_exp = df_rand_exp[df_rand_exp["Trial"] <= 100]
        df_rand_exp["Mode"] = "Experimental"

        df_sim_logei = helpers.load_data(
            # path / "sim_logei",
            path / "icassp26_sim_logei",
            f"*100-{n_init}*.npz",
            common.SEARCH_SPACE,
            common.TRUE_SIM_VALUES,
            include_time=False,
        )
        df_sim_logei["Mode"] = "Simulated"

        df_exp_logei = helpers.load_data(
            # path / "exp_logei",
            path / "icassp26_exp_logei",
            f"*100-{n_init}*.npz",
            common.SEARCH_SPACE,
            common.TRUE_EXP_VALUES,
            include_time=False,
        )
        df_exp_logei["Mode"] = "Experimental"
        print(len(df_sim_logei))
        print(len(df_exp_logei))

        df_sim_turbo_b1 = helpers.load_data(
            path / "icassp26_sim_turbo",
            "*.npz",
            common.SEARCH_SPACE,
            common.TRUE_SIM_VALUES,
            include_time=False,
        )
        df_sim_turbo_b1["Strategy"] = "TuRBO (q=1)"
        df_sim_turbo_b1["Mode"] = "Simulated"

        df_exp_turbo_b1 = helpers.load_data(
            path / "icassp26_exp_turbo",
            "*.npz",
            common.SEARCH_SPACE,
            common.TRUE_SIM_VALUES,
            include_time=False,
        )
        df_exp_turbo_b1["Strategy"] = "TuRBO (q=1)"
        df_exp_turbo_b1["Mode"] = "Experimental"

        df_sim_turbo_b4 = helpers.load_data(
            path / "icassp26_sim_turbo_b4",
            "*.npz",
            common.SEARCH_SPACE,
            common.TRUE_SIM_VALUES,
            include_time=False,
        )
        df_sim_turbo_b4["Strategy"] = "TuRBO (q=4)"
        df_sim_turbo_b4["Mode"] = "Simulated"

        df_exp_turbo_b4 = helpers.load_data(
            path / "icassp26_exp_turbo_b4",
            "*.npz",
            common.SEARCH_SPACE,
            common.TRUE_SIM_VALUES,
            include_time=False,
        )
        df_exp_turbo_b4["Strategy"] = "TuRBO (q=4)"
        df_exp_turbo_b4["Mode"] = "Experimental"

        df = pd.concat(
            [
                df_rand_sim,
                df_rand_exp,
                df_sim_logei,
                df_exp_logei,
                df_sim_turbo_b1,
                df_exp_turbo_b1,
                df_sim_turbo_b4,
                df_exp_turbo_b4,
            ]
        )
        df.to_csv(common.SWELLEX96Paths.outputs / "icassp26_results.csv", index=False)
    else:
        df = pd.read_csv(common.SWELLEX96Paths.outputs / "icassp26_results.csv")

    # Plot objective values and regret
    fig = performance_plot(df)
    fig.savefig(
        Path.cwd().parent / "reports" / "manuscripts" / "icassp26" / "figure2.png",
        **SAVEFIG_KWARGS,
    )

    # Plot final parameter estimates
    fig = plot_parameter_estimates(
        [
            df[
                (df["Mode"] == "Simulated")
                & (df["Strategy"] == "TuRBO (q=1)")
                & (df["Trial"] == 100)
            ],
            df[
                (df["Mode"] == "Experimental")
                & (df["Strategy"] == "TuRBO (q=1)")
                & (df["Trial"] == 100)
            ],
        ],
        parameters=[common.TRUE_SIM_VALUES, common.TRUE_EXP_VALUES],
    )
    fig.savefig(
        Path.cwd().parent / "reports" / "manuscripts" / "icassp26" / "figure3.png",
        **SAVEFIG_KWARGS,
    )

    # Print error matrices
    get_err_matrix(df, mode="Simulated")
    get_err_matrix(df, mode="Experimental")


if __name__ == "__main__":
    main(n_init=N_INIT)
