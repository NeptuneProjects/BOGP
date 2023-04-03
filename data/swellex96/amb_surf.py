#!/usr/bin/env python3

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# from figures import plot_ambiguity_surface
from tritonoa.at.models.kraken.runner import run_kraken
from tritonoa.sp.noise import added_wng, beamformer, snrdb_to_sigma


CBAR_KWARGS = {"location": "right", "pad": 0, "shrink": 1.0}
DATADIR = Path.cwd() / "Data" / "SWELLEX96" / "VLA" / "selected"
zvec = np.linspace(1, 200, 100)
rvec = np.linspace(10e-3, 10, 500)
NT = 350
SKIP_T = (
    list(range(73, 85))
    + list(range(95, 103))
    + list(range(187, 199))
    + list(range(287, 294))
    + list(range(302, 309))
)


def simulate_measurement_covariance(env_parameters: dict, snr: float = np.inf) -> np.array:
    sigma = snrdb_to_sigma(snr)
    p = run_kraken(env_parameters)
    p /= np.linalg.norm(p)
    noise = added_wng(p.shape, sigma=sigma, cmplx=True)
    p += noise
    K = p.dot(p.conj().T)
    return K



# B = np.abs(amb_surf) / np.max(np.abs(amb_surf))
# B = 10 * np.log10(B)
B = amb_surf

figpath = savepath.parent / "figures"
os.makedirs(figpath, exist_ok=True)

fig = plt.figure(figsize=(8, 6), facecolor="w", dpi=200)
ax, im = plot_ambiguity_surface(
    B, rvec, zvec, cmap="viridis", vmin=0, vmax=1
)
ax.axvline(ranges[t], color="r")
cbar = plt.colorbar(im, ax=ax, **CBAR_KWARGS)
cbar.set_label("Normalized Correlation")
ax.set_xlabel("Range [km]")
ax.set_ylabel("Depth [m]")
ax.set_title(f"Time Step = {t + 1:03d}, GPS Range = {ranges[t]:.2f} km")
fig.savefig(figpath / f"ambsurf_t={t + 1:03d}.png")
plt.close()
