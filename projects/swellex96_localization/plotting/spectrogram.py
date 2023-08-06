#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plot spectrogram of a .npz file.
Usage:
python3 ./data/plotting/spectrogram.py \
    ../data/swellex96_S5_VLA/acoustic/processed/merged.npz \
    --channel 10 \
    --fs 1500 \
    --nperseg 1024 \
    --show
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from typing import Optional

from tritonoa.plotting import plot_spectrogram


@dataclass
class Paths:
    source: Path
    destination: Path


@dataclass
class SpectrogramParameters:
    fs: float = 1500
    nperseg: int = 1024
    noverlap: Optional[int] = None
    nfft: Optional[int] = None


def compute_spectrogram(
    X: np.ndarray, spec_params: SpectrogramParameters
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f, t, Sxx = signal.spectrogram(
        X,
        spec_params.fs,
        nperseg=spec_params.nperseg,
        noverlap=spec_params.noverlap,
        nfft=spec_params.nfft,
    )
    Sxx_norm = Sxx / np.max(Sxx)
    Sxx_dB = 10 * np.log10(Sxx_norm)
    return f, t, Sxx_dB


def plot(data: dict, channel: int, spec_params: SpectrogramParameters) -> plt.Figure:
    X = data["X"]

    f, t, Sxx_dB = compute_spectrogram(X[..., channel], spec_params)

    fig = plt.figure(figsize=(8, 6), facecolor="w", dpi=125)
    _, im = plot_spectrogram(Sxx_dB, t, f, kwargs={"vmin": -40, "vmax": 0})
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    plt.title(f"Channel {channel}")
    cbar = plt.colorbar(im)
    cbar.set_label("Power [dB]")
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plots spectrogram of a .npz file.")
    parser.add_argument("path", type=Path, help="Path to data file.")
    parser.add_argument(
        "--fs", default=1500, type=float, help="Sampling frequency [Hz]."
    )
    parser.add_argument("--channel", default=0, type=int, help="Channel to plot.")
    parser.add_argument("--nperseg", default=1024, type=int, help="Nperseg.")
    parser.add_argument("--noverlap", default=None, type=int, help="Noverlap.")
    parser.add_argument("--nfft", default=None, type=int, help="Nfft.")
    parser.add_argument(
        "--save",
        default=Path("spectrogram.png"),
        type=Path,
        help="Path to save figure.",
    )
    parser.add_argument(
        "--show", action=argparse.BooleanOptionalAction, help="Show figure."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = Paths(source=args.path, destination=args.path.parent / args.save)
    spec_params = SpectrogramParameters(
        fs=args.fs, nperseg=args.nperseg, noverlap=args.noverlap, nfft=args.nfft
    )

    fig = plot(
        data=np.load(paths.source),
        channel=args.channel,
        spec_params=spec_params,
    )
    if args.save:
        fig.savefig(paths.destination, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
