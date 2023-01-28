#!/usr/bin/env python3

from argparse import ArgumentParser

import numpy as np
from scipy.fft import fft

from tritonoa.io import SIODataHandler


def covariance(p):
    d = np.expand_dims(p, 1)
    d /= np.linalg.norm(d)
    return d.dot(d.conj().T)


def find_freq_bin(fvec, X, f0):
    f_lower = f0 - 1
    f_upper = f0 + 1
    ind = (fvec >= f_lower) & (fvec < f_upper)
    data = (np.abs(X).sum(axis=1) / X.shape[1])
    data[~ind] = -2009
    return np.argmax(data)


def process_data(datadir, destination):
    x, _ = SIODataHandler.load_merged(datadir)

    fs = 1500
    M = x.shape[1]
    NT = 350
    N_snap = x.shape[0] // NT
    NFFT = 2 ** 13
    freq = 232

    fvec = (fs / NFFT) * np.arange(0, NFFT)

    # x[:, 42] = 0 # Remove corrupted channel
    x[:, 42] = x[:, [41,43]].mean(axis=1)

    p = np.zeros((NT, M), dtype=complex)
    for i in range(NT):
        idx_start = i * N_snap
        idx_end = (i + 1) * N_snap

        X = fft(x[idx_start:idx_end], n=NFFT, axis=0)
        # X = fftshift(X)
        fbin = find_freq_bin(fvec, X, freq)
        # print(fvec[fbin])
        p[i] = X[fbin]

    
    np.save(destination, p)

    # for i in range(NT):
    #     K = covariance(p[i])
    
    
    return

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--datadir", type=str, default="/Users/williamjenkins/Research/Projects/BOGP/Data/SWELLEX96/VLA/selected/merged.npz")
    parser.add_argument("--destination", type=str, default="/Users/williamjenkins/Research/Projects/BOGP/Data/SWELLEX96/VLA/selected/data.npy")
    args = parser.parse_args()
    process_data(args.datadir, args.destination)
