#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from conf.common import SWELLEX96Paths

TIME_STEP = 121
NOISE_FREQS = [143, 248, 401]
SIG_FREQS = [148, 235, 388]

snr_path = SWELLEX96Paths.acoustic_path / "snr"


def main() -> None:
    for nfreq, sfreq in zip(NOISE_FREQS, SIG_FREQS):
        noise = np.real(
            np.diag(np.load(snr_path / f"{nfreq:1d}.0Hz" / "covariance.npy")[TIME_STEP])
        )
        signal = np.real(
            np.diag(np.load(snr_path / f"{sfreq:1d}.0Hz" / "covariance.npy")[TIME_STEP])
        )
        snr = 10 * np.log10(np.mean(signal - noise) / np.mean(noise))
        print(f"SNR for {sfreq} Hz: {snr:.2f} dB")


if __name__ == "__main__":
    main()
