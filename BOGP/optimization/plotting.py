#!/usr/bin/env python3

from matplotlib import cm, colors
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch

from .optimizer import import_from_str, ObjectiveFunction

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


def plot_best_observations(
    best_values,
    labels: list = None,
    ax=None,
    lcb_constraint=None,
    ucb_constraint=None,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()

    for i in range(len(best_values)):
        data = best_values[i]
        mu = data.mean(axis=0)
        sigma = data.std(axis=0)
        _plot_best_observations(mu, ax=ax, kwargs={"label": labels[i]})
        _plot_best_observations_confint(
            mu,
            sigma,
            ax=ax,
            lcb_constraint=lcb_constraint,
            ucb_constraint=ucb_constraint,
            kwargs={"alpha": 0.1},
        )

    ax.legend()
    ax.set_xlabel("Observation")
    ax.set_ylabel("Best Observed Value")

    return ax


def _plot_best_observations(best_values, ax=None, kwargs={}):
    if ax is None:
        ax = plt.gca()
    return ax.plot(np.arange(len(best_values)), best_values, **kwargs)


def _plot_best_observations_confint(
    mu, sigma, ax=None, lcb_constraint=None, ucb_constraint=None, kwargs={}
):
    if ax is None:
        ax = plt.gca()

    lcb = mu - sigma
    if lcb_constraint is not None:
        lcb[lcb < lcb_constraint] = lcb_constraint
    ucb = mu + sigma
    if ucb_constraint is not None:
        ucb[ucb > ucb_constraint] = ucb_constraint

    ax = ax.fill_between(np.arange(len(mu)), lcb, ucb, **kwargs)
    return ax


def plot_performance_history(df, ax=None, label=None):
    if ax is None:
        ax = plt.gca()

    mean = df.mean(axis=0).values
    std = df.std(axis=0).values
    ucb = mean + std
    lcb = mean - std
    evals = df.columns.to_numpy()

    line, = ax.plot(evals, mean, label=label)
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
    ):
        if ObjFunc is None:
            ObjFunc = import_from_str(
                self.optim["obj_func_module"], self.optim["obj_func_name"]
            )
        if len(parameters_to_plot) == 1:
            return self._plot_training_1D(X_test, ObjFunc, index)
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

    def _plot_training_1D(self, X_test, ObjFunc, index):

        parameters = {
            item["name"]: X_test[..., i]
            for i, item in enumerate(self.optim["search_parameters"])
        }
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
                ax.set_title(f"Index {index[i]}")
            ax.set_ylabel("Objective\nFunction")
            ax.set_xlim([X_test.min() - X_offset, X_test.max() + X_offset])

            if i != len(results) - 1:
                alpha = self._evaluate_acq_func(result.acqfunc, X_test.unsqueeze(-1))
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
                ax.set_xlabel("Domain")
                ax.set_xlim([X_test.min() - X_offset, X_test.max() + X_offset])

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
