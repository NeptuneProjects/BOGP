#!/usr/bin/env python3

from argparse import ArgumentParser
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.io import savemat
from tqdm.auto import tqdm

from figures import plot_ambiguity_surface
from tritonoa.io import SIODataHandler
from tritonoa.kraken import run_kraken
from tritonoa.sp import beamformer
import swellex

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


def covariance(p):
    d = np.expand_dims(p, 1)
    d /= np.linalg.norm(d)
    return d.dot(d.conj().T)


def find_freq_bin(fvec, X, f0):
    f_lower = f0 - 1
    f_upper = f0 + 1
    ind = (fvec >= f_lower) & (fvec < f_upper)
    data = np.abs(X).sum(axis=1) / X.shape[1]
    data[~ind] = -2009
    return np.argmax(data)


def process_data(datadir, destination, freqs):
    print(f"Processing frequencies: {freqs} Hz")
    destination = Path(destination) if isinstance(destination, str) else destination
    x, _ = SIODataHandler.load_merged(datadir)
    x[:, 42] = x[:, [41, 43]].mean(axis=1)  # Remove corrupted channel
    x = np.fliplr(x)  # Reverse channel index

    for freq in tqdm(freqs, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"):

        fs = 1500
        M = x.shape[1]
        NT = 350
        N_snap = x.shape[0] // NT
        NFFT = 2**13

        fvec = (fs / NFFT) * np.arange(0, NFFT)

        p = np.zeros((NT, M), dtype=complex)
        f_hist = np.zeros(NT)
        for i in range(NT):
            idx_start = i * N_snap
            # idx_end = (i + 1) * N_snap
            idx_end = idx_start + NFFT

            X = fft(x[idx_start:idx_end], n=NFFT, axis=0)
            fbin = find_freq_bin(fvec, X, freq)
            f_hist[i] = fvec[fbin]
            p[i] = X[fbin]

        savepath = destination / f"{freq:.1f}Hz"
        os.makedirs(savepath, exist_ok=True)
        np.save(savepath / "data.npy", p)
        np.save(savepath / "f_hist.npy", f_hist)
        savemat(savepath / f"data_{freq}Hz.mat", {"p": p, "f": freq})


def generate_ambiguity_surfaces(datadir, freqs, overwrite=False):

    datadir = Path(datadir) if isinstance(datadir, str) else datadir
    environment = swellex.environment
    ranges = pd.read_csv(datadir / "gps_range.csv")["Range [km]"].values

    for f in freqs:
        p = np.load(datadir / f"{f:.1f}Hz" / "data.npy")
        # p = np.fliplr(p)

        for t in tqdm(
            range(NT),
            # range(250, 350),
            # [249],
            desc=f"Processing {f} Hz",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
            unit="time step",
        ):
            if t in SKIP_T:
                continue

            savepath = datadir / f"{f:.1f}Hz" / f"ambsurf_t={t + 1:03d}.npy"

            if savepath.exists() and overwrite:
                continue

            d = np.expand_dims(p[t], 1)
            d /= np.linalg.norm(d)
            K = d.dot(d.conj().T)

            amb_surf = np.zeros((len(zvec), len(rvec)))
            for zz, z in enumerate(zvec):
                p_rep = run_kraken(
                    environment
                    | {
                        "src_z": z,
                        "rec_r": rvec,
                        "freq": f,
                        "tilt": -1,
                        "tmpdir": ".",
                    }
                )

                for rr, r in enumerate(rvec):
                    amb_surf[zz, rr] = beamformer(K, p_rep[:, rr], atype="cbf").item()

            np.save(savepath, amb_surf)

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


def multifreq(datadir, freqs):

    ranges = pd.read_csv(datadir / "gps_range.csv")["Range [km]"].values

    for t in range(NT):
        surfs = np.zeros((len(freqs), len(zvec), len(rvec)))
        savepath = datadir / "multifreq" / "".join([f"{f}-" for f in freqs])[:-1]
        os.makedirs(savepath, exist_ok=True)
        try:
            for ff, f in enumerate(freqs):
                loadpath = datadir / f"{f:.1f}Hz" / f"ambsurf_t={t + 1:03d}.npy"
                surf = np.load(loadpath)
                surfs[ff] = surf

            B = np.mean(surfs, axis=0)
            np.save(savepath / f"ambsurf_mf_t={t + 1:03d}.npy", B)

            figpath = savepath / "figures"
            os.makedirs(figpath, exist_ok=True)

            fig = plt.figure(figsize=(8, 6), facecolor="w", dpi=200)
            # ax, im = plot_ambiguity_surface(B, rvec, zvec, cmap="cmo.thermal", vmin=-20)
            ax, im = plot_ambiguity_surface(B, rvec, zvec, vmin=0, vmax=1)
            ax.axvline(ranges[t], color="r")
            cbar = plt.colorbar(im, ax=ax, **CBAR_KWARGS)
            cbar.set_label("Centered Average Correlation [dB re max(B)]")
            ax.set_xlabel("Range [km]")
            ax.set_ylabel("Depth [m]")
            ax.set_title(
                f"Multi-Freq {freqs} Hz\nTime Step = {t + 1:03d}, GPS Range = {ranges[t]:.2f} km, Max Corr = {B.max():.2f}"
            )
            fig.savefig(figpath / f"ambsurf_t={t + 1:03d}.png")
            plt.close()

        except FileNotFoundError:
            pass


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("operation", type=str)
    # parser.add_argument(
    #     "--datadir",
    #     type=str,
    #     default="/Users/williamjenkins/Research/Projects/BOGP/Data/SWELLEX96/VLA/selected/merged.npz",
    # )
    parser.add_argument(
        "--destination",
        type=str,
        default=DATADIR,
    )
    parser.add_argument("--freqs", type=str, default="all")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.freqs != "all":
        freqs = list(map(lambda i: float(i.strip()), args.freqs.split(",")))
    else:
        freqs = swellex.HIGH_SIGNAL_TONALS

    if args.operation == "process":
        datadir = DATADIR / "merged.npz"
        process_data(datadir, args.destination, freqs)
    elif args.operation == "ambsurf":
        datadir = DATADIR
        generate_ambiguity_surfaces(datadir, freqs, args.overwrite)
    elif args.operation == "multifreq":
        datadir = DATADIR
        multifreq(datadir, freqs)
