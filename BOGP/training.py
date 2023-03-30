#!/usr/bin/env python3

from pathlib import Path

from botorch import fit_gpytorch_model, fit_gpytorch_mll
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.models import SingleTaskGP
from botorch.test_functions import Griewank
from gpytorch.constraints import GreaterThan, LessThan, Interval, Positive
from gpytorch.kernels import MaternKernel, ProductKernel, RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from acoustics import simulate_measurement_covariance
from figures import load_training_data
import swellex
from tritonoa.kraken import run_kraken
from tritonoa.sp import beamformer


class MatchedFieldProcessor:
    """For use in BoTorch training example."""

    def __init__(self, K, frequencies, parameters, atype="cbf"):
        self.K = K
        self.frequencies = frequencies
        self.parameters = parameters
        self.atype = atype

    def __call__(self, parameters):
        return self.evaluate(parameters)

    def evaluate(self, parameters):
        B = []
        for f, k in zip(self.frequencies, self.K):
            p_rep = run_kraken(self.parameters | {"freq": f} | parameters)
            B.append(beamformer(k, p_rep, atype=self.atype).item())

        return np.mean(np.array(B))


class ObjectiveFunction:
    def __init__(self, obj_func):
        self.obj_func = obj_func

    def evaluate(
        self,
        parameters: dict,
        fixed_parameters: dict = {},
        dtype=torch.double,
        disable_pbar=True,
    ):
        num_samples = set([len(param) for param in parameters.values()])
        if not len(num_samples) == 1:
            raise ValueError("The number of samples is inconsistent across features.")
        y = []
        pbar = tqdm(
            range(next(iter(num_samples))),
            desc="Evaluating test points",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
            leave=True,
            position=0,
            unit=" eval",
            disable=disable_pbar,
            colour="red",
        )
        for i in pbar:
            params = {
                k: float(v[i].detach().cpu().numpy()) for k, v in parameters.items()
            }
            y.append(self.obj_func(fixed_parameters | params))
        return torch.Tensor(y).unsqueeze(-1).to(dtype)


def get_bounds(search_parameters):
    bounds = torch.zeros(2, len(search_parameters))
    for i, parameter in enumerate(search_parameters):
        bounds[:, i] = torch.tensor(parameter["bounds"])
    return bounds


def get_test_points(bounds, num_samples, dtype=torch.double):
    d = bounds.size(1)
    if d == 1:
        bounds = bounds.squeeze()
        X = torch.linspace(bounds[0], bounds[1], num_samples).unsqueeze(-1).to(dtype)
    else:
        if isinstance(num_samples, int):
            num_samples = [num_samples for _ in range(d)]
        x = []
        for i in range(d):
            x.append(
                torch.linspace(bounds[0, i], bounds[1, i], num_samples[i]).to(dtype)
            )

        Xm = torch.meshgrid(*x, indexing="xy")
        X = torch.cat([x.flatten().unsqueeze(-1) for x in Xm], dim=1)
    return X


def generate_random_data(bounds, num_samples, dtype=torch.double):
    d = bounds.size(1)
    if d == 1:
        bounds = bounds.squeeze()
        X = (
            torch.distributions.uniform.Uniform(bounds[0], bounds[1])
            .sample([num_samples])
            .unsqueeze(-1)
            .to(dtype)
        )
    else:
        X_tmp = []
        for i in range(d):
            X_tmp.append(
                torch.distributions.uniform.Uniform(bounds[0, i], bounds[1, i])
                .sample([num_samples])
                .to(dtype)
            )
        X = torch.cat([x.flatten().unsqueeze(-1) for x in X_tmp], dim=1)
    return X


def generate_gridded_data(bounds, num_samples, dtype=torch.double):
    d = bounds.size(1)
    if d == 1:
        bounds = bounds.squeeze()
        X = torch.linspace(bounds[0], bounds[1], num_samples).unsqueeze(-1).to(dtype)
    else:
        if isinstance(num_samples, int):
            num_samples = [num_samples for _ in range(d)]
        x = []
        for i in range(d):
            x.append(
                torch.linspace(bounds[0, i], bounds[1, i], num_samples[i]).to(dtype)
            )

        Xm = torch.meshgrid(*x, indexing="xy")
        X = torch.cat([x.flatten().unsqueeze(-1) for x in Xm], dim=1)
    return X


def generate_initial_data(
    bounds,
    obj_func,
    search_space,
    fixed_parameters,
    n_train=10,
    n_test=None,
    dtype=torch.double,
):
    # X_train = generate_random_data(bounds, n_train, dtype)
    X_train = generate_gridded_data(bounds, [10, 5], dtype)
    train_parameters = {
        item["name"]: X_train[..., i] for i, item in enumerate(search_space)
    }
    y_train = obj_func.evaluate(train_parameters, fixed_parameters)
    best_y = y_train.max()

    if n_test is not None:
        X_test = get_test_points(bounds, n_test)
        test_parameters = {
            item["name"]: X_test[..., i] for i, item in enumerate(search_space)
        }
        y_test = obj_func.evaluate(
            test_parameters, fixed_parameters, disable_pbar=False
        )
    else:
        X_test, y_test = None, None

    return X_train, y_train, best_y, X_test, y_test
 

def initialize_model(X_train, y_train, state_dict=None):

    # kernel_range = RBFKernel(active_dims=(0), lengthscale_constraint=Interval(0.03, 0.3))
    # kernel_range = MaternKernel(active_dims=(0), lengthscale_constraint=GreaterThan(0.1))
    # kernel_range = MaternKernel(active_dims=(0))
    # kernel_depth = RBFKernel(active_dims=(1), lengthscale_constraint=Interval(0.02, 0.2))
    # kernel_depth = MaternKernel(active_dims=(1), lengthscale_constraint=GreaterThan(5))
    # kernel_depth = MaternKernel(active_dims=(1))
    # base_kernel = ProductKernel(kernel_range, kernel_depth)
    # base_kernel = MaternKernel(ard_num_dims=2)
    base_kernel = RBFKernel(ard_num_dims=X_train.shape[-1])

    model = SingleTaskGP(
        train_X=X_train,
        train_Y=y_train,
        mean_module=ConstantMean(constant_constraint=Positive()),
        covar_module=ScaleKernel(
            base_kernel=base_kernel
        )
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    if state_dict is not None:
        model.load_state_dict(state_dict)
        
    # print(model.covar_module.base_kernel.kernels._modules['0'].lengthscale)
    # print(model.covar_module.base_kernel.kernels._modules['1'].lengthscale)
    print(model.covar_module.base_kernel.lengthscale)
    # print(model.covar_module.lengthscale)
    
    return mll, model


def optimize_acqf_and_get_observation(
    X, acq_func, search_space, obj_func, fixed_parameters
):
    alpha = acq_func(X.unsqueeze(1))
    ind = torch.argmax(alpha)
    X_new = X[ind].detach().unsqueeze(-1).T
    new_parameters = {
        item["name"]: X_new[..., i] for i, item in enumerate(search_space)
    }
    y_new = obj_func.evaluate(new_parameters, fixed_parameters)
    return X_new, y_new, alpha


def _optimize_acqf_and_get_observation(
    X, acq_func, search_space, obj_func, fixed_parameters
):
    bounds = get_bounds(search_space)
    candidates, _ = optimize_acqf(
        acq_function=acq_func, bounds=bounds, q=1, num_restarts=40, raw_samples=1024
    )
    alpha = acq_func(X.unsqueeze(1))
    X_new = candidates.detach()
    new_parameters = {
        item["name"]: X_new[..., i] for i, item in enumerate(search_space)
    }
    y_new = obj_func.evaluate(new_parameters, fixed_parameters)

    return X_new, y_new, alpha


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


def main(optimization):
    seed = 2009
    dtype = torch.double
    torch.manual_seed(seed)
    env_parameters = swellex.environment | {"tmpdir": "."}

    true_parameters = {
        "rec_r": 1.0,
        "src_z": 60,
    }
    # frequencies = [201]
    frequencies = [148, 166, 201, 235, 283, 338, 388]
    # frequencies = [49, 64, 79, 94, 112, 130, 148, 166, 201, 235, 283, 338, 388]
    K = []
    for f in frequencies:
        K.append(
            simulate_measurement_covariance(
                env_parameters | {"snr": 20, "freq": f} | true_parameters
            )
        )
    K = np.array(K)
    if len(K.shape) == 2:
        K = K[np.newaxis, ...]
    obj_func = ObjectiveFunction(MatchedFieldProcessor(K, frequencies, env_parameters))

    if optimization == "r":
        savepath = Path.cwd() / "Data" / "range_estimation" / "demo2"
        search_space = [
            {
                "name": "rec_r",
                "type": "range",
                "bounds": [0.1, 8.0],
                "value_type": "float",
                "log_scale": False,
            }
        ]
        NUM_TRIALS = 50
        n_test = 501
        n_train = 3
        bounds = get_bounds(search_space)
        X_train, y_train, best_y, _, _ = generate_initial_data(
            bounds, obj_func, search_space, env_parameters, n_train=n_train, dtype=dtype
        )
        X_test = get_test_points(bounds, num_samples=n_test)
        X_t = X_test.detach().cpu().numpy()
        rvec = np.unique(X_t[:, 0])
        zvec = [true_parameters["src_z"]]
    elif optimization == "l":
        savepath = Path.cwd() / "Data" / "localization" / "demo2"
        search_space = [
            {
                "name": "rec_r",
                "type": "range",
                "bounds": [0.1, 4.0],
                "value_type": "float",
                "log_scale": False,
            },
            {
                "name": "src_z",
                "type": "range",
                "bounds": [1.0, 160.0],
                "value_type": "float",
                "log_scale": False,
            },
        ]
        NUM_TRIALS = 50
        n_test = [100, 100]
        n_train = 50
        bounds = get_bounds(search_space)
        X_train, y_train, best_y, _, _ = generate_initial_data(
            bounds, obj_func, search_space, env_parameters, n_train=n_train, dtype=dtype
        )
        X_test = get_test_points(bounds, num_samples=n_test)
        X_t = X_test.detach().cpu().numpy()
        rvec = np.unique(X_t[:, 0])
        zvec = np.unique(X_t[:, 1])

    pbar = tqdm(
        total=len(rvec) * len(zvec) * len(frequencies),
        bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
        leave=True,
        position=0,
        unit=" depths",
        colour="red",
    )

    B = np.zeros((len(frequencies), len(zvec), len(rvec)))
    for ff, f in enumerate(frequencies):
        for zz, z in enumerate(zvec):
            p_rep = run_kraken(
                env_parameters | {"freq": f, "src_z": z, "rec_r": rvec}
            )
            for rr, _ in enumerate(rvec):
                B[ff, zz, rr] = beamformer(K[ff], p_rep[:, rr], atype="cbf").item()
                pbar.update(1)
    pbar.close()
    B = np.mean(B, axis=0)

    y_actual = torch.from_numpy(B.flatten()).to(dtype)

    mll, model = initialize_model(X_train, y_train)

    pbar = tqdm(
        range(NUM_TRIALS),
        desc="Optimizing",
        bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
        leave=True,
        position=0,
        unit=" eval",
        colour="blue",
    )

    mean = np.zeros((NUM_TRIALS, np.prod(n_test)))
    ucb = np.zeros_like(mean)
    lcb = np.zeros_like(mean)
    alpha_array = np.zeros_like(mean)

    for trial in pbar:
        # fit_gpytorch_model(mll)
        fit_gpytorch_mll(mll)

        mll.eval()
        with torch.no_grad():
            posterior = mll.model(X_test)
            y_test = posterior.mean
            cov_test = posterior.confidence_region()

            mean[trial] = y_test.detach().cpu().numpy()
            lcb[trial] = cov_test[0].detach().cpu().numpy()
            ucb[trial] = cov_test[1].detach().cpu().numpy()

        EI = ExpectedImprovement(model=model, best_f=best_y)
        X_new, y_new, alpha = _optimize_acqf_and_get_observation(
            X_test, EI, search_space, obj_func, env_parameters
        )

        alpha_array[trial] = alpha.detach().cpu().numpy()

        X_train = torch.cat([X_train, X_new])
        y_train = torch.cat([y_train, y_new])
        best_y = y_train.max()

        mll, model = initialize_model(X_train, y_train, model.state_dict())

    X_test_array = X_test.detach().cpu().numpy()
    y_actual_array = y_actual.detach().cpu().numpy()
    X_train_array = X_train.detach().cpu().numpy()
    y_train_array = y_train.detach().cpu().numpy()

    np.save(savepath / "X_test.npy", X_test_array)
    np.save(savepath / "y_actual.npy", y_actual_array)
    np.save(savepath / "X_train.npy", X_train_array)
    np.save(savepath / "y_train.npy", y_train_array)
    np.save(savepath / "mean.npy", mean)
    np.save(savepath / "lcb.npy", lcb)
    np.save(savepath / "ucb.npy", ucb)
    np.save(savepath / "alpha.npy", alpha_array)


def plot_training(
    X_test,
    y_actual,
    X_train,
    y_train,
    mean,
    lcb,
    ucb,
    alpha,
    alpha_prev=None,
    title=None,
    xlabel=None,
    ylabel=None,
    ylim=[0, 1],
):
    d = X_test.shape[1]

    if d == 1:
        fig, axs = plt.subplots(
            nrows=2, facecolor="white", figsize=(8, 6), gridspec_kw={"hspace": 0}
        )

        ax = axs[0]
        ax.set_title(title)

        ax = plot_gp_1D(
            X_test, y_actual, X_train, y_train, mean, lcb, ucb, alpha, alpha_prev, ax=ax
        )
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 5:
            del handles[4], labels[4]
            handles[-1] = Line2D([0], [0], color="r", marker="x", linestyle=":")
        ax.set_xticklabels([])
        ax.set_xlabel(None)
        ax.set_ylim(ylim)
        ax.set_ylabel("$f(\mathbf{X})$", rotation=0, ha="right")

        ax = axs[1]
        ax = plot_acqf_1D(X_test, alpha, alpha_prev, ax=ax)

        handles2, labels2 = ax.get_legend_handles_labels()
        ax.legend(
            handles + handles2,
            labels + labels2,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            ncols=4,
        )

        ax.set_xlabel(xlabel)
        ax.set_ylim(ylim)
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("$\\alpha(\mathbf{X})$", rotation=0, ha="left")

        return fig
    elif d == 2:
        IMSHOW_KW = {
            "aspect": "auto",
            "origin": "lower",
            "interpolation": "none",
            "extent": [
                min(X_test[:, 0]),
                max(X_test[:, 0]),
                min(X_test[:, 1]),
                max(X_test[:, 1]),
            ],
        }
        ACTUAL_KW = {
            "marker": "s",
            "facecolors": "none",
            "edgecolors": "w",
            "zorder": 30,
        }
        NEXT_KW = {"facecolors": "none", "edgecolors": "r", "zorder": 60}
        SCATTER_KW = {"c": "w", "marker": "o", "zorder": 40, "alpha": 0.5, "s": 1}
        M = len(np.unique(X_test[:, 0]))
        N = len(np.unique(X_test[:, 1]))

        max_alpha, max_alpha_prev = get_candidates(
            np.reshape(alpha, (M, N)),
            np.reshape(alpha_prev, (M, N)) if alpha_prev is not None else None,
        )
        max_alpha = np.unravel_index(max_alpha, alpha.shape)

        if max_alpha_prev is not None:
            max_alpha_prev = np.unravel_index(max_alpha_prev, alpha_prev.shape)

        max_f, _ = get_candidates(np.reshape(y_actual, (M, N)))
        max_f = np.unravel_index(max_f, y_actual.shape)

        fig, axs = plt.subplots(
            ncols=4, facecolor="white", figsize=(16, 4), gridspec_kw={"wspace": 0}
        )

        ax = axs[0]
        ax.set_title("Actual")
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
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax = axs[1]
        ax.set_title("Mean Function")
        ax.imshow(np.reshape(mean, (M, N)), **IMSHOW_KW)
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

        ax = axs[2]
        ax.set_title("Covariance Function")
        ax.imshow(np.reshape(ucb, (M, N)), **IMSHOW_KW)
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

        ax = axs[3]
        ax.set_title("Acquisition Function")
        im = ax.imshow(np.reshape(alpha, (M, N)), **IMSHOW_KW)
        if max_alpha is not None:
            ax.scatter(*X_test[max_alpha], **NEXT_KW)
        ax.invert_yaxis()
        ax.set_xticklabels([])
        ax.set_xlabel(None)
        ax.set_yticklabels([])
        ax.set_ylabel(None)

        cax = ax.inset_axes([1.05, 0, 0.08, 1.0])
        fig.colorbar(im, ax=ax, cax=cax)

        fig.suptitle(title)

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


def plot_acqf_1D(X_test, alpha, alpha_prev=None, ax=None):
    if ax is None:
        ax = plt.gca()

    max_alpha, max_alpha_prev = get_candidates(alpha, alpha_prev)

    ax.plot(X_test, alpha, color="tab:red", label="$\\alpha(\mathbf{X})$")
    ax.axvline(X_test[max_alpha], color="k", linestyle="-")
    if max_alpha_prev:
        ax.axvline(X_test[max_alpha_prev], color="r", linestyle=":")
    return ax


def save_figs(optimization):
    if optimization == "r":
        loadpath = Path.cwd() / "Data" / "range_estimation" / "demo2"
        xlabel = "$\mathbf{X}=R_{src}$ [km]"
        ylabel = None
    elif optimization == "l":
        loadpath = Path.cwd() / "Data" / "localization" / "demo2"
        xlabel = "$R_{src}$ [km]"
        ylabel = "$z_{src}$ [m]"

    X_test, y_actual, X_train, y_train, mean, lcb, ucb, alpha = load_training_data(
        loadpath
    )

    trials = mean.shape[0]
    num_rand = X_train.shape[0] - mean.shape[0]

    for trial in range(trials):
        if trial == 0:
            fig = plot_training(
                X_test,
                y_actual,
                X_train[0 : num_rand + trial],
                y_train[0 : num_rand + trial],
                mean[trial],
                lcb[trial],
                ucb[trial],
                alpha[trial],
                title="Initialization",
                ylim=[-0.1, 1.1],
                xlabel=xlabel,
                ylabel=ylabel,
            )
        else:
            fig = plot_training(
                X_test,
                y_actual,
                X_train[0 : num_rand + trial],
                y_train[0 : num_rand + trial],
                mean[trial],
                lcb[trial],
                ucb[trial],
                alpha[trial],
                alpha[trial - 1],
                title=f"Iteration {trial}",
                ylim=[-0.1, 1.1],
                xlabel=xlabel,
                ylabel=ylabel,
            )
        fig.savefig(
            loadpath / "figures" / f"trial{trial:03d}.png", bbox_inches="tight", dpi=250
        )
        plt.close()


if __name__ == "__main__":
    optimization = "l"
    main(optimization)
    save_figs(optimization)
