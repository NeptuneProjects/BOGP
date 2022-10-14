#!/usr/bin/env python3
"""Usage:
python3 ./Source/scripts/figures.py
"""
import argparse
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from tritonoa.kraken import run_kraken
from tritonoa.io import read_ssp
from tritonoa.sp import beamformer, snrdb_to_sigma, added_wng

sys.path.insert(0, str(Path.cwd() / "Source"))
from BOGP import utils

ROOT = Path("~/Research/Projects/BOGP")
FIGURE_PATH = ROOT / "Reports" / "JASA" / "figures"


def main(figures: list):
    for i, _ in enumerate(figures):
        print(f"Producing Figure {i + 1}" + 60*'-')
        try:
            eval(f"figure{i + 1}()")
        except NameError:
            warnings.warn(f"Figure {i + 1} is not defined or implemented yet.")
            continue
            


def figure1():
    # Set the true parameters:
    range_true = 3.0
    depth_true = 62
    freq = 201

    # Load CTD data
    z_data, c_data, _ = read_ssp(ROOT / "Data" / "SWELLEX96" / "CTD" / "i9606.prn", 0, 3, header=None)
    z_data = np.append(z_data, 217)
    c_data = np.append(c_data, c_data[-1])

    fixed_parameters = {
        # General
        "title": "SWELLEX96_SIM",
        "tmpdir": ".",
        "model": "KRAKENC",
        # Top medium
        # Layered media
        "layerdata": [
            {
                "z": z_data,
                "c_p": c_data,
                "rho": 1
            },
            {
                "z": [217, 240],
                "c_p": [1572.37, 1593.02],
                "rho": 1.8,
                "a_p": 0.3
            },
            {
                "z": [240, 1040],
                "c_p": [1881, 3245.8],
                "rho": 2.1,
                "a_p": 0.09
            }
        ],
        # Bottom medium
        "bot_opt": "A",
        "bot_c_p": 5200,
        "bot_rho": 2.7,
        "bot_a_p": 0.03,
        # Speed constraints
        "clow": 0,
        "chigh": 1600,
        # Receiver parameters
        "rec_z": np.linspace(94.125, 212.25, 64),
        "rec_r": range_true,
        "snr": np.Inf,
        # Source parameters
        "src_z": depth_true,
        "freq": freq
    }

    # Run simulation with true parameters:
    sigma = snrdb_to_sigma(fixed_parameters["snr"])
    p_rec = run_kraken(fixed_parameters)
    p_rec /= np.linalg.norm(p_rec)
    noise = added_wng(p_rec.shape, sigma=sigma, cmplx=True)
    p_rec += noise
    K = p_rec.dot(p_rec.conj().T)

    # Define parameter search space:
    [fixed_parameters.pop(item) for item in ["rec_r", "src_z"]]
    search_parameters = [
        {
            "name": "rec_r",
            "bounds": [0.001, 12.0]
        },
        {
            "name": "src_z",
            "bounds": [0.5, 200.]
        }
    ]

    # Define search grid & run MFP
    dr = 1 / 1e3 # [km]
    rvec = np.arange(search_parameters[0]["bounds"][0], search_parameters[0]["bounds"][1] + dr, dr)
    zvec = np.arange(search_parameters[1]["bounds"][0], search_parameters[1]["bounds"][1], 1)

    pbar = tqdm(
        zvec,
        bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}',
        desc="  MFP",
        leave=True,
        position=0,
        unit=" step"
    )
    p_rep = np.zeros((len(zvec), len(rvec), len(fixed_parameters["rec_z"])))
    B_bart = np.zeros((len(zvec), len(rvec)))

    for zz, z in enumerate(pbar):
        p_rep = run_kraken(fixed_parameters | {"src_z": z, "rec_r": rvec})
        for rr, r in enumerate(rvec):
            B_bart[zz, rr] = beamformer(K, p_rep[:, rr], atype="cbf").item()

    utils.clean_up_kraken_files(".")

    # Bn = B_bart / np.max(B_bart)
    Bn = B_bart
    logBn = 10 * np.log10(Bn)
    src_z_ind, src_r_ind = np.unravel_index(np.argmax(logBn), (len(zvec), len(rvec)))

    fig = plt.figure(figsize=(6, 4))
    plt.imshow(
        logBn,
        aspect="auto",
        extent=[min(rvec), max(rvec), min(zvec), max(zvec)],
        origin="lower",
        vmin=-10, vmax=0, interpolation="none",
        cmap="jet"
    )
    plt.plot(rvec[src_r_ind], zvec[src_z_ind], 'w*', markersize=15, markeredgewidth=1.5, markeredgecolor="k")
    plt.gca().invert_yaxis()
    plt.xlabel("Range [km]")
    plt.ylabel("Depth [m]")
    plt.title("Ambiguity Surface")
    plt.colorbar(label="Normalized Correlation [dB]")
    fig.savefig(FIGURE_PATH / "figure01.png", dpi=300, facecolor="white")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("figures", type=str)
    args = parser.parse_args()
    figures = list(map(lambda i:int(i.strip()), args.figures.split(',')))
    main(figures)