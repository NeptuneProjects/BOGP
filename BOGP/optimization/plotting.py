#!/usr/bin/env python3

from matplotlib import cm, colors
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch

from .optimizer import import_from_str, ObjectiveFunction

SMOKE_TEST = False
SURF_KWARGS = {""}
CBAR_KWARGS = {"location": "top", "pad": 0, "shrink": 0.9}
LINE_KWARGS = {"c": "k", "ls": "dashed", "lw": 0.5}
SAMPLE_KWARGS = {"c": "k", "s": 2}
NEXT_KWARGS = {"c": "r", "marker": "X", "s": 50, "linewidth": 0.5, "edgecolor": "k"}
OPTIMUM_KWARGS = {
    "c": "#35fc03",
    "marker": "*",
    "s": 50,
    "linewidth": 0.5,
    "edgecolor": "k",
}


def plot_agg_data(
    df,
    simulations,
    value_to_plot,
    compute_error_with=None,
    optimum=None,
    upper_threshold=None,
    lower_threshold=None,
    ax=None
):
    if ax is None:
        ax = plt.gca()
    max_evals = 0
    lines = []
    for k, acq_func in enumerate(simulations["acq_func"]):
        selection = df["acq_func"] == acq_func
        if compute_error_with is not None:
            actual_value = (
                float(compute_error_with)
                if isinstance(compute_error_with, list)
                else compute_error_with
            )
            selected_data = abs(
                (
                    df[selection]
                    .pivot(index="seed", columns="evaluation", values=value_to_plot)
                    .astype(float)
                )
                - actual_value
            )
        else:
            selected_data = (
                df[selection]
                .pivot(index="seed", columns="evaluation", values=value_to_plot)
                .astype(float)
            )

        if len(selected_data.columns) > max_evals:
            max_evals = len(selected_data.columns)
        line = plot_line_with_confint(
            selected_data,
            ax=ax,
            label=simulations["acq_func_abbrev"][k],
            upper_threshold=upper_threshold,
            lower_threshold=lower_threshold,
        )
        lines.append(line)

    if optimum is not None:
        optimum = ax.axhline(
            1.0, c="k", ls="--", lw=1, alpha=0.5, label="Global Optimum"
        )
        lines.append(optimum)

    return lines


def plot_ambiguity_surface(
    B,
    rvec,
    zvec,
    ax=None,
    cmap="jet",
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

    Bn = B
    logBn = 10 * np.log10(Bn)
    src_z_ind, src_r_ind = np.unravel_index(np.argmax(logBn), (len(zvec), len(rvec)))

    im = ax.imshow(
        logBn,
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

    return ax, im


def plot_aggregated_data(
    df,
    evaluations,
    value_to_plot,
    compute_error_with=None,
    optimum=None,
    upper_threshold=None,
    lower_threshold=None,
    xlabel="x",
    ylabel="y",
    xlim=None,
    ylim=None,
    title=None,
):
    # TODO: Remove noise-free case
    """
    values: best_value, best_param, best_param0, best_param1, etc.
    """
    fig = plt.figure(figsize=(7, 7), facecolor="w")
    gs = GridSpec(4, 2, figure=fig, hspace=0.075, wspace=0.05)  # Grid is hard-coded

    lines = []
    legend_populated = False
    max_evals = 0
    for i, row in enumerate(evaluations["rec_r"]):
        for j, col in enumerate(evaluations["snr"]):
            ax = fig.add_subplot(gs[i, j])
            for k, acq_func in enumerate(evaluations["acq_func"]):
                selection = (
                    (df["acq_func"] == acq_func)
                    & (df["snr"] == float(col))
                    & (df["rec_r"] == float(row))
                )

                if compute_error_with is not None:
                    selected_data = abs(
                        (
                            df[selection]
                            .pivot(
                                index="seed", columns="evaluation", values=value_to_plot
                            )
                            .astype(float)
                        )
                        - (
                            float(compute_error_with[i])
                            if isinstance(compute_error_with, list)
                            else compute_error_with
                        )
                    )
                else:
                    selected_data = (
                        df[selection]
                        .pivot(index="seed", columns="evaluation", values=value_to_plot)
                        .astype(float)
                    )

                if len(selected_data.columns) > max_evals:
                    max_evals = len(selected_data.columns)
                line = plot_line_with_confint(
                    selected_data,
                    label=evaluations["acq_func_abbrev"][k],
                    upper_threshold=upper_threshold,
                    lower_threshold=lower_threshold,
                )
                if not legend_populated:
                    lines.append(line)

            if optimum is not None:
                optimum = ax.axhline(
                    1.0, c="k", ls="--", lw=1, alpha=0.5, label="Global Optimum"
                )
                if not legend_populated:
                    lines.append(optimum)

            legend_populated = True

            if xlim is None:
                ax.set_xlim([-5, max_evals + 10])
            else:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

            if i == 0 and j == 0:
                ax.set_title("Noiseless")
            elif i == 0 and j == 1:
                ax.set_title(f"SNR = {col} dB")

            if i != 3:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])

            if j == 0:
                if i != 3:
                    ax.set_ylabel(f"$R={float(row):.1f}$ km\n")
                elif i == 3:
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(f"$R={float(row):.1f}$ km\n{ylabel}")

            if i == 3 and j == 1:
                ax.legend(
                    lines,
                    [l.get_label() for l in lines],
                    loc="right",
                    ncol=3 if len(lines) % 3 == 0 else 2,
                    bbox_to_anchor=(1, -0.4),
                )

    fig.suptitle(title, y=0.94)

    return fig


def plot_line_with_confint(
    df, ax=None, label=None, lower_threshold=None, upper_threshold=None
):
    if ax is None:
        ax = plt.gca()

    mean = df.mean(axis=0).values
    std = df.std(axis=0).values

    ucb = mean + std
    if upper_threshold is not None:
        ucb[ucb > upper_threshold] = upper_threshold
    lcb = mean - std
    if lower_threshold is not None:
        lcb[lcb < lower_threshold] = lower_threshold
    evals = df.columns.to_numpy()

    (line,) = ax.plot(evals, mean, label=label)
    ax.fill_between(evals, lcb, ucb, alpha=0.2)
    return line


class ResultsPlotter:
    def __init__(self, optim: dict, results: list):
        self.optim = optim
        self.results = results

    def __del__(self):
        pass

    def plot_training_iterations(
        self,
        X_test,
        parameters_to_plot,
        M=101,
        N=101,
        index=None,
        parameter_labels=None,
        reverse_x=False,
        reverse_y=False,
        ObjFunc=None,
        objfunc_kwargs=None,
        consolidate=False,
        savepath=None,
        show=False,
    ):
        if ObjFunc is None:
            ObjFunc = import_from_str(
                self.optim["obj_func_module"], self.optim["obj_func_name"]
            )
        if len(parameters_to_plot) == 1:
            return self._plot_training_1D(
                X_test,
                ObjFunc,
                index,
                parameter_labels=parameter_labels,
                consolidate=consolidate,
                savepath=savepath,
                show=show,
            )
        elif len(parameters_to_plot) == 2:
            return self._plot_training_2D(
                X_test,
                ObjFunc,
                parameters_to_plot,
                M,
                N,
                index,
                parameter_labels,
                reverse_x,
                reverse_y,
            )
        else:
            raise NotImplementedError("A maximum of two parameters may be plotted!")

    def _plot_training_1D(
        self,
        X_test,
        ObjFunc,
        index,
        parameter_labels=None,
        consolidate=False,
        savepath=None,
        show=False,
    ):
        parameters = {
            item["name"]: X_test[..., i]
            for i, item in enumerate(self.optim["search_parameters"])
        }
        if not SMOKE_TEST:
            y_test = self._evaluate_obj_func(
                ObjFunc,
                parameters,
                self.optim["fixed_parameters"],
                self.optim["obj_func_kwargs"],
            )
        X_offset = 0.1 * X_test.min()

        if index is not None:
            results = [result for i, result in enumerate(self.results) if i in index]
        else:
            results = self.results

        if consolidate:

            fig = plt.figure(figsize=(6, 12), facecolor="white")
            outer_grid = fig.add_gridspec(len(results), 1, hspace=0.1)

            for i, result in enumerate(results):
                mu, variance = self._evaluate_posterior(result.model.posterior, X_test)

                inner_grid = outer_grid[i].subgridspec(
                    2, 1, hspace=0, height_ratios=[3, 2]
                )
                axs = inner_grid.subplots()
                if not SMOKE_TEST:
                    axs[0] = self._plot_gp_1D(
                        result.X.detach().cpu().numpy().squeeze(),
                        result.y.detach().cpu().numpy().squeeze(),
                        result.X_new.detach().cpu().numpy().squeeze()
                        if result.X_new is not None
                        else None,
                        X_test.detach().cpu().numpy().squeeze(),
                        y_test.detach().cpu().numpy().squeeze(),
                        mu.detach().cpu().numpy().squeeze(),
                        variance.detach().cpu().numpy().squeeze(),
                        axs[0],
                    )
                axs[0].set_xticks([])
                if i == 0:
                    axs[0].text(0.1, 0.9, "Initialization", va="top")
                elif index is None:
                    axs[0].text(0.1, 0.9, f"Iteration {i}", va="top")
                else:
                    axs[0].text(0.1, 0.9, f"Iteration {index[i]}", va="top")
                axs[0].set_xlim([X_test.min() - X_offset, X_test.max() + X_offset])
                axs[0].set_yticks([0.0, 1.0])
                # if i == len(results) - 1:
                axs[0].set_ylabel("$f(\mathbf{x})$", rotation=0, va="center")
                axs[0].yaxis.set_label_coords(-0.05, 0.5)

                # if i != len(results) - 1:
                if not SMOKE_TEST:
                    alpha = self._evaluate_acq_func(
                        result.acqfunc, X_test.unsqueeze(-1)
                    )

                    axs[1] = self._plot_acqfunc_1D(
                        X_test.detach().cpu().numpy().squeeze(),
                        result.X_new.detach().cpu().numpy().squeeze()
                        if result.X_new is not None
                        else None,
                        (alpha / alpha.abs().max()).detach().cpu().numpy().squeeze()
                        if alpha is not None
                        else None,
                        axs[1],
                    )
                axs[1].set_xlim([X_test.min() - X_offset, X_test.max() + X_offset])
                axs[1].yaxis.tick_right()
                axs[0].set_yticks([0.0, 1.0])
                # if i == len(results) - 1:
                axs[1].set_ylabel("$\\alpha(\mathbf{x})$", rotation=0, va="center")
                axs[1].yaxis.set_label_coords(1.05, 0.5)

                if i == len(results) - 1:
                    if parameter_labels is None:
                        axs[1].set_xlabel("x1")
                    else:
                        axs[1].set_xlabel(parameter_labels)
                else:
                    axs[1].set_xticklabels([])

            if savepath is not None:
                fig.savefig(savepath / f"1D_training_{i:03d}.png", bbox_inches="tight")
            if show:
                plt.show()
            return fig

        else:

            for i, result in enumerate(results):
                mu, variance = self._evaluate_posterior(result.model.posterior, X_test)

                fig = plt.figure(figsize=(6, 4), facecolor="white")
                gspec = GridSpec(2, 1, hspace=0.0, height_ratios=[3, 2])

                ax = fig.add_subplot(gspec[0])
                ax = self._plot_gp_1D(
                    result.X.detach().cpu().numpy().squeeze(),
                    result.y.detach().cpu().numpy().squeeze(),
                    result.X_new.detach().cpu().numpy().squeeze()
                    if result.X_new is not None
                    else None,
                    X_test.detach().cpu().numpy().squeeze(),
                    y_test.detach().cpu().numpy().squeeze(),
                    mu.detach().cpu().numpy().squeeze(),
                    variance.detach().cpu().numpy().squeeze(),
                    ax,
                )
                ax.set_xticks([])
                if index is None:
                    ax.set_title(f"Iteration {i}")
                else:
                    ax.set_title(f"Iteration {index[i]}")
                ax.set_ylabel("Objective\nFunction")
                ax.set_xlim([X_test.min() - X_offset, X_test.max() + X_offset])

                if i != len(results) - 1:
                    alpha = self._evaluate_acq_func(
                        result.acqfunc, X_test.unsqueeze(-1)
                    )
                    ax = fig.add_subplot(gspec[1])
                    ax = self._plot_acqfunc_1D(
                        X_test.detach().cpu().numpy().squeeze(),
                        result.X_new.detach().cpu().numpy().squeeze()
                        if result.X_new is not None
                        else None,
                        alpha.detach().cpu().numpy().squeeze()
                        if alpha is not None
                        else None,
                        ax,
                    )
                    ax.set_ylabel("Acquisition\nFunction")
                    if parameter_labels is not None:
                        ax.set_xlabel("x1")
                    else:
                        ax.set_xlabel(parameter_labels)
                    ax.set_xlim([X_test.min() - X_offset, X_test.max() + X_offset])

            if savepath is not None:
                fig.savefig(savepath / f"1D_training_{i:03d}.png")

            if show:
                plt.show()

    def _plot_training_2D(
        self,
        X_test,
        ObjFunc,
        parameters_to_plot,
        M,
        N,
        index,
        parameter_labels=None,
        reverse_x=False,
        reverse_y=False,
    ):
        # Locate the indexes of which parameters are to be plotted.
        idx_to_plot = self._get_param_index(
            self.optim["search_parameters"], parameters_to_plot
        )
        idx1 = idx_to_plot[0]
        idx2 = idx_to_plot[1]
        # Evalaute the objective function at the test points.
        parameters = {
            item: X_test.squeeze()[..., i] for i, item in enumerate(parameters_to_plot)
        }
        y_test = (
            self._evaluate_obj_func(
                ObjFunc,
                parameters,
                self.optim["fixed_parameters"],
                self.optim["obj_func_kwargs"],
            )
            .detach()
            .cpu()
            .numpy()
            .squeeze()
            .reshape(M, N)
        )
        vmin_f, vmax_f = y_test.min(), y_test.max()
        x1_test = X_test[:, :, idx1].detach().cpu().numpy().squeeze().reshape(M, N)
        x2_test = X_test[:, :, idx2].detach().cpu().numpy().squeeze().reshape(M, N)

        # Keep only the results specified by plotting index
        if index is not None:
            results = [result for i, result in enumerate(self.results) if i in index]
        else:
            results = self.results

        for i, result in enumerate(results):
            # Get the mean and variance from the posterior at the test points.
            mu, variance = self._evaluate_posterior(result.model.posterior, X_test)
            mu = mu.detach().cpu().numpy().squeeze().reshape(M, N)
            variance = variance.detach().cpu().numpy().squeeze().reshape(M, N)
            # Get the candidate points suggested by the acquisition function.
            x1_new = (
                result.X_new[:, idx1].detach().cpu().numpy().squeeze()
                if result.X_new is not None
                else None
            )
            x2_new = (
                result.X_new[:, idx2].detach().cpu().numpy().squeeze()
                if result.X_new is not None
                else None
            )
            # Get the previous sample locations.
            x1 = result.X[:, idx1].detach().cpu().numpy().squeeze()
            x2 = result.X[:, idx2].detach().cpu().numpy().squeeze()

            fig = plt.figure(figsize=(15, 3), facecolor="white")
            gspec = GridSpec(1, 5, wspace=0.0)

            # Plot objective function
            ax = fig.add_subplot(gspec[0])
            ax = self._plot_surf_2D(
                x1_test,
                x2_test,
                y_test,
                x1_new=x1_new,
                x2_new=x2_new,
                ax=ax,
                surf_kwargs={"cmap": cm.cividis, "vmin": vmin_f, "vmax": vmax_f},
                cbar_kwargs=CBAR_KWARGS | {"label": r"$f$"},
                line_kwargs=LINE_KWARGS,
                next_kwargs=NEXT_KWARGS,
                optimum_kwargs=OPTIMUM_KWARGS,
            )
            if reverse_x:
                ax.invert_xaxis()
            if reverse_y:
                ax.invert_yaxis()
            if parameter_labels is None:
                ax.set_xlabel(r"$x_1$")
                ax.set_ylabel(r"$x_2$")
            else:
                ax.set_xlabel(parameter_labels[0])
                ax.set_ylabel(parameter_labels[1])

            # Plot mean function
            ax = fig.add_subplot(gspec[1])
            ax = self._plot_surf_2D(
                x1_test,
                x2_test,
                mu,
                x1_new=x1_new,
                x2_new=x2_new,
                x1_sample=x1,
                x2_sample=x2,
                ax=ax,
                surf_kwargs={"cmap": cm.cividis, "vmin": vmin_f, "vmax": vmax_f},
                cbar_kwargs=CBAR_KWARGS | {"label": r"$\mu$"},
                line_kwargs=LINE_KWARGS,
                sample_kwargs=SAMPLE_KWARGS,
                next_kwargs=NEXT_KWARGS,
                optimum_kwargs=OPTIMUM_KWARGS,
            )
            if reverse_x:
                ax.invert_xaxis()
            if reverse_y:
                ax.invert_yaxis()
            ax.set_xticklabels([])
            ax.set_yticks([])

            # Plot error surface
            ax = fig.add_subplot(gspec[2])
            ax = self._plot_surf_2D(
                x1_test,
                x2_test,
                mu - y_test,
                x1_new=x1_new,
                x2_new=x2_new,
                x1_sample=x1,
                x2_sample=x2,
                ax=ax,
                surf_kwargs={"cmap": cm.RdBu},
                cbar_kwargs=CBAR_KWARGS | {"label": r"$\mu - f$"},
                line_kwargs=LINE_KWARGS,
                sample_kwargs=SAMPLE_KWARGS,
                show_next_sample=False,
                show_optimum=False,
            )
            if reverse_x:
                ax.invert_xaxis()
            if reverse_y:
                ax.invert_yaxis()
            ax.set_xticklabels([])
            ax.set_yticks([])

            # Plot variance
            ax = fig.add_subplot(gspec[3])
            ax = self._plot_surf_2D(
                x1_test,
                x2_test,
                variance,
                x1_new=x1_new,
                x2_new=x2_new,
                x1_sample=x1,
                x2_sample=x2,
                ax=ax,
                surf_kwargs={"cmap": cm.RdYlGn_r},
                cbar_kwargs=CBAR_KWARGS | {"label": r"$\sigma$"},
                line_kwargs=LINE_KWARGS,
                sample_kwargs=SAMPLE_KWARGS,
                show_next_sample=False,
                show_optimum=False,
            )
            if reverse_x:
                ax.invert_xaxis()
            if reverse_y:
                ax.invert_yaxis()
            ax.set_xticklabels([])
            ax.set_yticks([])

            # Plot acquisition function
            if i != len(results) - 1:
                alpha = (
                    self._evaluate_acq_func(result.acqfunc, X_test)
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze()
                    .reshape(M, N)
                )
                alpha[alpha < 0.0] = 0.0

                ax = fig.add_subplot(gspec[4])
                ax = self._plot_surf_2D(
                    x1_test,
                    x2_test,
                    alpha,
                    x1_new=x1_new,
                    x2_new=x2_new,
                    ax=ax,
                    # surf_kwargs={"cmap": cm.plasma, "norm": colors.LogNorm()},
                    surf_kwargs={"cmap": cm.plasma},
                    cbar_kwargs=CBAR_KWARGS | {"label": r"$\alpha$"},
                    line_kwargs=LINE_KWARGS,
                    sample_kwargs=SAMPLE_KWARGS,
                    show_next_sample=False,
                    show_optimum=False,
                )
                if reverse_x:
                    ax.invert_xaxis()
                if reverse_y:
                    ax.invert_yaxis()
                ax.set_xticklabels([])
                ax.set_yticks([])

                plt.suptitle(
                    f"Evaluation {i + 1}: Estimated (True) Range = {self.results[-1].best_parameters[0]:.2f} ({self.optim['obj_func_kwargs']['parameters']['rec_r']}) | Depth = {self.results[-1].best_parameters[1]:.2f} ({self.optim['obj_func_kwargs']['parameters']['src_z']})"
                )

            plt.show()

        return ax

    @staticmethod
    def _evaluate_acq_func(acqfunc, X):
        with torch.no_grad():
            return acqfunc(X)

    @staticmethod
    def _evaluate_obj_func(
        ObjFunc, parameters: dict, fixed_parameters: dict = {}, obj_func_kwargs={}
    ):

        obj_func = ObjectiveFunction(ObjFunc(**obj_func_kwargs))
        return obj_func.evaluate(parameters, fixed_parameters)

    @staticmethod
    def _evaluate_posterior(posterior, X):
        posterior = posterior(X, require_grad=False)
        mu = posterior.mean.detach().cpu()
        variance = posterior.variance.detach().cpu()
        return mu, variance

    @staticmethod
    def _get_param_index(search_parameters, parameters_to_plot):
        return [
            i
            for i, item in enumerate(search_parameters)
            if item["name"] in parameters_to_plot
        ]

    @staticmethod
    def _plot_gp_1D(X, y, X_new, X_test, y_test, mu, variance, ax=None):
        if ax is None:
            ax = plt.gca()

        ucb = mu + 2 * np.sqrt(variance)
        lcb = mu - 2 * np.sqrt(variance)

        ax.plot(X_test, mu, label="Mean")
        ax.scatter(X, y, label="Samples", c="k", marker="x", zorder=10)
        ax.fill_between(X_test, ucb, lcb, alpha=0.2, label="2*std")

        if X_new is not None:
            try:
                for item in X_new:
                    ax.axvline(x=item, c="k", ls="dashed")
            except TypeError:
                ax.axvline(x=X_new, c="k", ls="dashed")

        ax.plot(X_test, y_test, label="True Objective", c="r")
        return ax

    @staticmethod
    def _plot_surf_2D(
        x1,
        x2,
        y,
        x1_new=None,
        x2_new=None,
        x1_sample=None,
        x2_sample=None,
        ax=None,
        surf_kwargs={},
        cbar_kwargs={},
        line_kwargs={},
        sample_kwargs={},
        next_kwargs={},
        optimum_kwargs={},
        show_next_sample=True,
        show_optimum=True,
    ):
        if ax is None:
            ax = plt.gca()
        im = ax.imshow(
            y.T,
            extent=[x1[:, 0].min(), x1[:, 0].max(), x2[0, :].min(), x2[0, :].max()],
            interpolation="none",
            aspect="auto",
            origin="lower",
            **surf_kwargs,
        )
        cbar = plt.colorbar(im, ax=ax, **cbar_kwargs)
        # Plot location of samples
        if (x1_sample is not None) and (x2_sample is not None):
            ax.scatter(x1_sample, x2_sample, **sample_kwargs)
        # Plot location of candidates
        if x1_new is not None:
            try:
                for item in x1_new:
                    ax.axvline(item, **line_kwargs)
            except TypeError:
                ax.axvline(x1_new, **line_kwargs)
        if x2_new is not None:
            try:
                for item in x2_new:
                    ax.axhline(item, **line_kwargs)
            except TypeError:
                ax.axhline(x2_new, **line_kwargs)
        if (x1_new is not None) and (x2_new is not None) and show_next_sample:
            ax.scatter(x1_new, x2_new, **next_kwargs)
        # Plot location of global optimum
        if show_optimum:
            ind1_max, ind2_max = np.unravel_index(np.argmax(y), y.shape)
            ax.scatter(x1[ind1_max, 0], x2[0, ind2_max], **optimum_kwargs)
        ax.set_xlim(x1[:, 0].min(), x1[:, 0].max())
        ax.set_ylim(x2[0, :].min(), x2[0, :].max())
        return ax

    @staticmethod
    def _plot_acqfunc_1D(X_test, X_new, alpha, ax=None):
        if ax is None:
            ax = plt.gca()

        ax.plot(X_test, alpha)
        try:
            for item in X_new:
                ax.axvline(x=item, c="k", ls="dashed")
        except TypeError:
            ax.axvline(x=X_new, c="k", ls="dashed")
        return ax
