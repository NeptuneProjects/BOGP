import time
import warnings

from botorch import fit_gpytorch_model
from botorch.acquisition.analytic import ExpectedImprovement, ProbabilityOfImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.test_functions import *
from gpytorch.mlls import ExactMarginalLogLikelihood
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double
torch.manual_seed(1)

DIM = 1
BATCH_SIZE = 1
NUM_RESTARTS = 10
RAW_SAMPLES = 512
N_TRIALS = 3
N_BATCH = 50
MC_SAMPLES = 256

# objective_func = Levy(dim=DIM, negate=True)
# bounds = torch.tensor([[-10.0] * DIM, [10.0] * DIM], device=device, dtype=dtype)
# objective_func = Ackley(dim=DIM, negate=True)
# bounds = torch.tensor([[-32.768] * DIM, [32.768] * DIM], device=device, dtype=dtype)
# objective_func = DixonPrice(dim=DIM, negate=True)
# bounds = torch.tensor([[-10.0] * DIM, [10.0] * DIM], device=device, dtype=dtype)
# objective_func = Griewank(dim=DIM, negate=True)
# bounds = torch.tensor([[-600.0] * DIM, [600.0] * DIM], device=device, dtype=dtype)
# objective_func = Michalewicz(dim=DIM, negate=True)
# bounds = torch.tensor([[0.0] * DIM, [torch.pi] * DIM], device=device, dtype=dtype)
objective_func = Rastrigin(dim=DIM, negate=True)
bounds = torch.tensor([[-5.12] * DIM, [5.12] * DIM], device=device, dtype=dtype)
# objective_func = StyblinskiTang(dim=DIM, negate=True)
# bounds = torch.tensor([[-5.] * DIM, [5.] * DIM], device=device, dtype=dtype)


def generate_initial_data(n=10, dim=2):
    train_X = (float(bounds[0]) - float(bounds[1])) * torch.rand(
        n, dim, device=device, dtype=dtype
    ) + float(bounds[1])
    train_y = objective_func(train_X).unsqueeze(-1)
    best_observed_value = train_y.max().item()
    return train_X, train_y, best_observed_value


def initialize_model(train_X, train_y, state_dict=None):
    model = SingleTaskGP(train_X, train_y).to(train_X)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


def optimize_acqf_and_get_observation(acq_func, q=1):
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=q,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        options={},
    )

    new_X = candidates.detach()
    new_y = objective_func(new_X).unsqueeze(-1)
    return new_X, new_y


def plot_gp_1D(train_X, train_y, new_X, posterior, objective, test_X, ax=None):

    mu = posterior.mean
    variance = posterior.variance
    ucb = mu + 2 * torch.sqrt(variance)
    lcb = mu - 2 * torch.sqrt(variance)

    actual_y = objective.evaluate_true(test_X).detach().numpy().flatten()

    if ax is None:
        ax = plt.gca()

    train_X = train_X.detach().numpy().flatten()
    train_y = train_y.detach().numpy().flatten()
    test_X = test_X.detach().numpy().flatten()
    mu = mu.detach().numpy().flatten()
    ucb = ucb.detach().numpy().flatten()
    lcb = lcb.detach().numpy().flatten()

    ax.plot(test_X, mu, label="Mean")
    ax.scatter(train_X, train_y, label="Samples", c="k", marker="x", zorder=10)
    ax.fill_between(test_X, ucb, lcb, alpha=0.2, label="2*std")
    for item in new_X.tolist():
        ax.axvline(x=item, c="k", ls="dashed")
    ax.plot(test_X, -actual_y, label="True Objective", c="r")

    return ax


def plot_acqfunc_1D(test_X, new_X, acqfunc, ax=None):
    with torch.no_grad():
        alpha = acqfunc(test_X[:, None, None]).detach().numpy().flatten()

    if ax is None:
        ax = plt.gca()

    ax.plot(test_X, alpha)
    # ax.axvline(new_X.detach().numpy(), c="k", ls="dashed")
    for item in new_X.tolist():
        ax.axvline(x=item, c="k", ls="dashed")
    return ax


def plot_training_1D(
    train_X, train_y, new_X, test_X, posterior, objective, acqfunc, title=None
):
    fig = plt.figure(figsize=(6, 4))
    gspec = GridSpec(2, 1, hspace=0.0, height_ratios=[3, 2])

    x_offset = 0.01 * test_X.min()
    ax0 = fig.add_subplot(gspec[0])
    ax0 = plot_gp_1D(train_X, train_y, new_X, posterior, objective, test_X, ax0)
    ax0.spines.bottom.set_visible(False)
    ax0.set_xticks([])
    ax0.set_title(title)
    ax0.set_ylabel("Objective\nFunction")
    ax0.set_xlim([test_X.min() - x_offset, test_X.max() + x_offset])

    ax1 = fig.add_subplot(gspec[1])
    ax1 = plot_acqfunc_1D(test_X, new_X, acqfunc, ax1)
    ax1.set_ylabel("Acquisition\nFunction")
    ax1.set_xlabel("Domain")
    ax1.set_xlim([test_X.min() - x_offset, test_X.max() + x_offset])

    # ax1.spines.top.set_visible(False)
    return fig


def plot_best_observed(best_observed, ax=None, label=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(best_observed, label=label)
    return ax


def plot_best_observations(best_observations, best_possible):
    fig = plt.figure()

    for label, best_observed in best_observations.items():
        plot_best_observed(best_observed, label=label)

    plt.axhline(best_possible, c="k", ls="dashed", label="Global Optimum")

    plt.xlabel("Number of Observations")
    plt.ylabel("Best Observed Objective Value")
    plt.legend()
    return fig


def optimization_loop():
    test_X = torch.linspace(float(bounds[0]), float(bounds[1]), 201).unsqueeze(-1)

    # EI
    best_observed_ei = []
    train_X_ei, train_y_ei, best_observed_value_ei = generate_initial_data(n=5, dim=DIM)
    mll_ei, model_ei = initialize_model(train_X_ei, train_y_ei)

    # PI
    best_observed_pi = []
    train_X_pi, train_y_pi = train_X_ei, train_y_ei
    best_observed_value_pi = best_observed_value_ei
    mll_pi, model_pi = initialize_model(train_X_pi, train_y_pi)

    # qEI
    best_observed_qei = []
    train_X_qei, train_y_qei = train_X_ei, train_y_ei
    best_observed_value_qei = best_observed_value_ei
    mll_qei, model_qei = initialize_model(train_X_qei, train_y_qei)

    best_observed_ei.append(best_observed_value_ei)
    best_observed_pi.append(best_observed_value_pi)
    best_observed_qei.append(best_observed_value_qei)

    for iteration in range(1, N_BATCH + 1):
        t0 = time.time()

        # Fit the models
        fit_gpytorch_model(mll_ei)
        fit_gpytorch_model(mll_pi)
        fit_gpytorch_model(mll_qei)

        # Define sampler for acq function modules
        qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)

        # Define acq functions
        EI = ExpectedImprovement(model=model_ei, best_f=train_y_ei.max())

        PI = ProbabilityOfImprovement(model=model_pi, best_f=train_y_pi.max())

        qEI = qExpectedImprovement(
            model=model_qei, best_f=train_y_qei.max(), sampler=qmc_sampler
        )

        # Optimize and get new observation
        new_X_ei, new_y_ei = optimize_acqf_and_get_observation(EI)
        new_X_pi, new_y_pi = optimize_acqf_and_get_observation(PI)
        new_X_qei, new_y_qei = optimize_acqf_and_get_observation(qEI, q=3)

        # Plot results:
        if (iteration == 1) or (not iteration % 10):
            fig = plot_training_1D(
                train_X_ei,
                train_y_ei,
                new_X_ei,
                test_X,
                model_ei.posterior(test_X, requires_grad=False),
                objective_func,
                EI,
                title="EI",
            )
            plt.show()
            fig = plot_training_1D(
                train_X_pi,
                train_y_pi,
                new_X_pi,
                test_X,
                model_pi.posterior(test_X, requires_grad=False),
                objective_func,
                PI,
                title="PI",
            )
            plt.show()
            fig = plot_training_1D(
                train_X_qei,
                train_y_qei,
                new_X_qei,
                test_X,
                model_qei.posterior(test_X, requires_grad=False),
                objective_func,
                qEI,
                title="qEI",
            )
            plt.show()

        # Update training points
        train_X_ei = torch.cat([train_X_ei, new_X_ei])
        train_y_ei = torch.cat([train_y_ei, new_y_ei])

        train_X_pi = torch.cat([train_X_pi, new_X_pi])
        train_y_pi = torch.cat([train_y_pi, new_y_pi])

        train_X_qei = torch.cat([train_X_qei, new_X_qei])
        train_y_qei = torch.cat([train_y_qei, new_y_qei])

        # Update progress
        best_value_ei = train_y_ei.max().item()
        best_value_pi = train_y_pi.max().item()
        best_value_qei = train_y_qei.max().item()

        best_observed_ei.append(best_value_ei)
        best_observed_pi.append(best_value_pi)
        best_observed_qei.append(best_value_qei)

        # Reinitialize models
        mll_ei, model_ei = initialize_model(
            train_X_ei, train_y_ei, model_ei.state_dict()
        )
        mll_pi, model_pi = initialize_model(
            train_X_pi, train_y_pi, model_pi.state_dict()
        )
        mll_qei, model_qei = initialize_model(
            train_X_qei, train_y_qei, model_qei.state_dict()
        )

        t1 = time.time()

        if verbose:
            print(
                f"\nBatch {iteration:>2}: best_value (EI, PI, qEI) = "
                f"({best_value_ei:>4.2f}, {best_value_pi:>4.2f}, {best_value_qei:>4.2f}), "
                f"time = {t1-t0:>4.2f}.",
                end="",
            )
        else:
            print(".", end="")
    return

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    verbose = True

    # best_observed_all_ei, best_observed_all_qei = [], []

    # Average over multiple trials
    # for trial in range(1, N_TRIALS + 1):

    optimization_loop()

    
