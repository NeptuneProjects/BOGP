#!/usr/bin/env python3
"""
1-D animation of TuRBO (Trust Region Bayesian Optimization).

Two panels per frame:
  top    — true function, GP mean, GP ±2σ band, observations, next point
  bottom — LogEI acquisition function (full domain dimmed, trust region solid)

Both panels shade and outline the current trust region.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import botorch
import gpytorch.settings as gpt_settings
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401 — registers style sheets
import torch
from botorch.acquisition import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

plt.style.use(["science", "ieee", "std-colors"])

DOMAIN = (0.0, 1.0)
N_TEST = 500
X_TEST = np.linspace(*DOMAIN, N_TEST)

TR_COLOR = "#4477AA"
NEXT_COLOR = "#EE6677"


# ---------------------------------------------------------------------------
# Test function
# ---------------------------------------------------------------------------


def true_function(x: np.ndarray) -> np.ndarray:
    """Synthetic 1-D objective with several local optima."""
    return np.sin(5.0 * np.pi * x) * (1.0 + 0.3 * np.cos(12.0 * np.pi * x))


Y_TRUE = true_function(X_TEST)
_TRUE_MIN_IDX = int(np.argmin(Y_TRUE))
X_TRUE_MIN = X_TEST[_TRUE_MIN_IDX]
Y_TRUE_MIN = Y_TRUE[_TRUE_MIN_IDX]


# ---------------------------------------------------------------------------
# TuRBO state (mirrors turbo.py)
# ---------------------------------------------------------------------------


@dataclass
class TurboState:
    dim: int = 1
    batch_size: int = 1
    length: float = 0.8
    length_min: float = 0.5**7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = 0  # set in __post_init__
    success_counter: int = 0
    success_tolerance: int = 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max(4.0 / self.batch_size, float(self.dim) / self.batch_size)
        )


def _update_state(state: TurboState, Y_next: torch.Tensor) -> TurboState:
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0

    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


# ---------------------------------------------------------------------------
# GP helpers
# ---------------------------------------------------------------------------


def _fit_gp(X: torch.Tensor, train_Y: torch.Tensor) -> SingleTaskGP:
    likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
    covar_module = ScaleKernel(
        MaternKernel(
            nu=2.5,
            ard_num_dims=1,
            lengthscale_constraint=Interval(0.005, 4.0),
        )
    )
    model = SingleTaskGP(X, train_Y, likelihood=likelihood, covar_module=covar_module)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    with gpt_settings.max_cholesky_size(float("inf")):
        fit_gpytorch_mll(mll)
    return model


def _trust_region(
    state: TurboState, X: torch.Tensor, train_Y: torch.Tensor
) -> tuple[float, float]:
    # In 1-D the lengthscale weights normalise to 1, so TR = x* ± L/2.
    x_center = X[train_Y.argmax(), 0].item()
    lo = float(np.clip(x_center - state.length / 2.0, *DOMAIN))
    hi = float(np.clip(x_center + state.length / 2.0, *DOMAIN))
    return lo, hi


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def run_turbo(n_init: int = 5, n_iter: int = 30, seed: int = 42) -> list[dict]:
    """Run TuRBO on the 1-D test function and return per-iteration frame data."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dtype = torch.double

    X = SobolEngine(1, scramble=True, seed=seed).draw(n_init).to(dtype=dtype)
    # Negate: TuRBO maximises internally, so Y = -f(x) converts to minimisation.
    Y = -torch.tensor(true_function(X.numpy()), dtype=dtype).squeeze(-1)  # [N]

    state = TurboState()
    state.best_value = Y.max().item()

    X_test_t = torch.tensor(X_TEST, dtype=dtype).unsqueeze(-1)  # [N_TEST, 1]

    frames: list[dict] = []

    with botorch.settings.validate_input_scaling(False):
        for i in range(n_iter):
            y_mean = Y.mean()
            y_std = Y.std().clamp(min=1e-6)
            train_Y = ((Y - y_mean) / y_std).unsqueeze(-1)  # [N, 1]

            model = _fit_gp(X, train_Y)
            tr_lb, tr_ub = _trust_region(state, X, train_Y)

            # GP predictions over full test grid
            model.eval()
            with torch.no_grad():
                post = model.posterior(X_test_t)
                mu_std = post.mean.squeeze().numpy()
                sigma_std = post.variance.squeeze().sqrt().numpy()

            # Back to original scale for display (negate: GP models -f, we display f).
            mu = -(mu_std * y_std.item() + y_mean.item())
            sigma = sigma_std * y_std.item()

            # LogEI on standardised model over full domain
            acqf = qLogExpectedImprovement(model, best_f=train_Y.max())
            with torch.no_grad():
                logei = acqf(X_test_t.unsqueeze(1)).numpy()  # [N_TEST]

            in_tr = (X_TEST >= tr_lb) & (X_TEST <= tr_ub)
            logei_tr = np.where(in_tr, logei, np.nan)

            # Optimise within trust region to find next point
            tr_bounds = torch.tensor([[tr_lb], [tr_ub]], dtype=dtype)
            with gpt_settings.max_cholesky_size(float("inf")):
                X_next, _ = optimize_acqf(
                    acqf,
                    bounds=tr_bounds,
                    q=1,
                    num_restarts=10,
                    raw_samples=256,
                )
            x_next = X_next.item()

            frames.append(
                {
                    "iteration": i + 1,
                    "X": X.numpy().flatten().copy(),
                    "Y": (-Y).numpy().copy(),  # display original f(x), not -f(x)
                    "mu": mu,
                    "sigma": sigma,
                    "logei": logei,
                    "logei_tr": logei_tr,
                    "tr_lb": tr_lb,
                    "tr_ub": tr_ub,
                    "x_next": x_next,
                    "tr_length": state.length,
                    "n_success": state.success_counter,
                    "n_failure": state.failure_counter,
                    "success_tolerance": state.success_tolerance,
                    "failure_tolerance": state.failure_tolerance,
                }
            )

            # Evaluate and append (negate to keep maximisation convention)
            y_next = -float(true_function(np.array([x_next]))[0])
            X = torch.cat([X, X_next.detach()], dim=0)
            Y = torch.cat([Y, torch.tensor([y_next], dtype=dtype)], dim=0)
            state = _update_state(state, torch.tensor([y_next], dtype=dtype))

            if state.restart_triggered:
                print(f"  Restart triggered at iteration {i + 1}")
                state = TurboState()
                state.best_value = Y.max().item()

    return frames


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------


def build_animation(frames: list[dict]) -> tuple[plt.Figure, animation.FuncAnimation]:
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(6, 3),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.0},
    )

    # Precompute fixed axis limits across all frames.
    top_lo = min(Y_TRUE.min(), min((f["mu"] - 2.0 * f["sigma"]).min() for f in frames))
    top_hi = max(Y_TRUE.max(), max((f["mu"] + 2.0 * f["sigma"]).max() for f in frames))
    top_pad = 0.1 * (top_hi - top_lo)
    TOP_YLIM = (top_lo - top_pad, top_hi + 4 * top_pad)  # extra headroom for legend

    bot_lo = min(np.nanmin(f["logei"]) for f in frames)
    bot_hi = max(np.nanmax(f["logei"]) for f in frames)
    bot_pad = 0.1 * (bot_hi - bot_lo)
    BOT_YLIM = (bot_lo - bot_pad, bot_hi + bot_pad)

    def _tr_lines(ax: plt.Axes, lo: float, hi: float) -> None:
        ax.axvspan(lo, hi, alpha=0.10, color=TR_COLOR, lw=0)
        ax.axvline(lo, color=TR_COLOR, ls="--", lw=0.7, alpha=0.7)
        ax.axvline(hi, color=TR_COLOR, ls="--", lw=0.7, alpha=0.7)

    def update(i: int) -> None:
        f = frames[i]
        ax_top.clear()
        ax_bot.clear()

        # ── top: GP surrogate ─────────────────────────────────
        _tr_lines(ax_top, f["tr_lb"], f["tr_ub"])

        ax_top.plot(X_TEST, Y_TRUE, "k-", lw=1.2, label=r"$f(x)$ (true)")
        ax_top.plot(
            X_TEST,
            f["mu"],
            color=TR_COLOR,
            ls="--",
            lw=1.0,
            label=r"GP mean $\mu(x)$",
        )
        ax_top.fill_between(
            X_TEST,
            f["mu"] - 2.0 * f["sigma"],
            f["mu"] + 2.0 * f["sigma"],
            alpha=0.20,
            color=TR_COLOR,
            label=r"$\mu \pm 2\sigma$",
        )
        ax_top.scatter(
            f["X"],
            f["Y"],
            c="k",
            s=18,
            zorder=5,
            label=f"Observations ($n={len(f['X'])}$)",
        )

        y_at_next = true_function(np.array([f["x_next"]]))[0]
        ax_top.axvline(f["x_next"], color=NEXT_COLOR, ls=":", lw=1.0)
        ax_top.scatter(
            [f["x_next"]],
            [y_at_next],
            color=NEXT_COLOR,
            marker="D",
            s=40,
            zorder=6,
            label=r"$x_{t+1}$",
        )

        # True global minimum
        ax_top.scatter(
            [X_TRUE_MIN],
            [Y_TRUE_MIN],
            color="k",
            marker="*",
            s=120,
            zorder=7,
            label=rf"True min $f^*={Y_TRUE_MIN:.5f}$",
        )

        # Best observed minimum so far
        best_idx = int(np.argmin(f["Y"]))
        ax_top.scatter(
            [f["X"][best_idx]],
            [f["Y"][best_idx]],
            facecolors="none",
            edgecolors="#CCBB44",
            marker="*",
            s=120,
            linewidths=1.2,
            zorder=7,
            label=rf"Best obs. $={f['Y'][best_idx]:.5f}$",
        )

        ax_top.set_xlim(DOMAIN)
        ax_top.set_ylim(*TOP_YLIM)
        ax_top.set_ylabel(r"$f(x)$")
        leg = ax_top.legend(
            loc="upper center",
            fontsize=6,
            ncol=3,
            frameon=True,
            framealpha=1.0,
            edgecolor="k",
        )
        leg.set_zorder(20)
        ax_top.set_title(
            rf"TuRBO — Trial {f['iteration']} $\;|\;$ "
            rf"$\Lambda = {f['tr_length']:.3f}$ $\;|\;$ "
            rf"succ. {f['n_success']}/{f['success_tolerance']}  "
            rf"fail. {f['n_failure']}/{f['failure_tolerance']}",
            fontsize=8,
        )

        # ── bottom: LogEI ─────────────────────────────────────
        _tr_lines(ax_bot, f["tr_lb"], f["tr_ub"])

        ax_bot.plot(
            X_TEST, f["logei"], color="0.65", lw=0.8, label="LogEI (full domain)"
        )
        ax_bot.plot(
            X_TEST, f["logei_tr"], color="k", lw=1.1, label="LogEI (trust region)"
        )
        ax_bot.axvline(
            f["x_next"], color=NEXT_COLOR, ls=":", lw=1.0, label=r"$x_{t+1}$"
        )

        ax_bot.set_xlim(DOMAIN)
        ax_bot.set_ylim(*BOT_YLIM)
        ax_bot.set_xlabel(r"$x$")
        ax_bot.set_ylabel(r"$\log \operatorname{EI}(x)$")

    anim = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1500, repeat=True
    )
    return fig, anim


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Simulating TuRBO on 1-D test function …")
    frames = run_turbo(n_init=4, n_iter=30, seed=0)
    print(f"Collected {len(frames)} frames.")

    print("Building animation …")
    fig, anim = build_animation(frames)

    out = Path.cwd().parent / "reports" / "figures" / "turbo_animation.gif"
    print(f"Saving → {out}")
    anim.save(str(out), writer="pillow", fps=1.0, dpi=300)
    print("Done.")
    # plt.show()
