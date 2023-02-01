#!/usr/bin/env python3

from argparse import ArgumentParser
import os
from pathlib import Path
from typing import List

import numpy as np
from scipy.fft import fft
from tqdm.auto import tqdm

from tritonoa.io import SIODataHandler
from tritonoa.kraken import run_kraken
from tritonoa.sp import beamformer
import swellex

HIGH_SIGNAL_TONALS = [
    49.0,
    64.0,
    79.0,
    94.0,
    112.0,
    130.0,
    148.0,
    166.0,
    201.0,
    235.0,
    283.0,
    338.0,
    388.0,
]


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

    for freq in tqdm(freqs, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"):

        fs = 1500
        M = x.shape[1]
        NT = 350
        N_snap = x.shape[0] // NT
        NFFT = 2**13

        fvec = (fs / NFFT) * np.arange(0, NFFT)

        # x[:, 42] = 0 # Remove corrupted channel
        x[:, 42] = x[:, [41, 43]].mean(axis=1)

        p = np.zeros((NT, M), dtype=complex)
        for i in range(NT):
            idx_start = i * N_snap
            idx_end = (i + 1) * N_snap

            X = fft(x[idx_start:idx_end], n=NFFT, axis=0)
            # X = fftshift(X)
            fbin = find_freq_bin(fvec, X, freq)
            # print(fvec[fbin])
            p[i] = X[fbin]

        savepath = destination / f"{freq:.1f}Hz"
        os.makedirs(savepath, exist_ok=True)
        np.save(savepath / "data.npy", p)


def generate_ambiguity_surfaces(datadir, freqs):

    datadir = Path(datadir) if isinstance(datadir, str) else datadir
    environment = swellex.environment
    NT = 350
    zvec = np.linspace(0, 200, 101)
    rvec = np.linspace(1e-3, 10 + 1e-3, 1001)

    for f in freqs:
        p = np.load(datadir / f"{f:.1f}Hz" / "data.npy")

        for t in tqdm(
            range(NT),
            desc=f"Processing {f} Hz",
            bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
            unit="time step",
        ):
            savepath = datadir / f"{f:.1f}Hz" / f"ambsurf_t={t + 1:03d}.npy"
            if savepath.exists():
                continue

            d = np.expand_dims(p[t], 1)
            d /= np.linalg.norm(d)
            K = d.dot(d.conj().T)
            
            amb_surf = np.zeros((len(zvec), len(rvec)))
            for zz, z in enumerate(zvec):
                p_rep = run_kraken(environment | {"src_z": z, "rec_r": rvec, "freq": f})
                for rr, r in enumerate(rvec):
                    amb_surf[zz, rr] = beamformer(K, p_rep[:, rr], atype="cbf").item()
                    
            
            np.save(savepath, amb_surf)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("operation", type=str)
    parser.add_argument(
        "--datadir",
        type=str,
        default="/Users/williamjenkins/Research/Projects/BOGP/Data/SWELLEX96/VLA/selected/merged.npz",
    )
    parser.add_argument(
        "--destination",
        type=str,
        default="/Users/williamjenkins/Research/Projects/BOGP/Data/SWELLEX96/VLA/selected",
    )
    parser.add_argument("--freqs", type=str, default="all")
    args = parser.parse_args()
    if args.freqs != "all":
        freqs = list(map(lambda i: float(i.strip()), args.freqs.split(",")))
    else:
        freqs = HIGH_SIGNAL_TONALS

    if args.operation == "process":
        process_data(args.datadir, args.destination, freqs)
    elif args.operation == "ambsurf":
        generate_ambiguity_surfaces(args.datadir, freqs)
