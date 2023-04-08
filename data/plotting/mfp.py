#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Usage:
python3 ./data/plotting/mfp.py \
    ../data/swellex96_S5_VLA/acoustic/ambiguity_surfaces/49-64-79-94-112-130-148-166-201-235-283-338-388_40x20 \
    --glob "*.npy"
"""

from argparse import ArgumentParser
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np

from tritonoa.plotting import plot_ambiguity_surface


def get_filenames(source_path: Path, glob: str):
    """Get all files matching the filename in the source path."""
    source_path = Path(source_path)
    return list(source_path.glob(glob))


def load_ambiguity_surface(filename):
    return np.load(filename)


def load_grid_data(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def plot(grid_data, ambiguity_surface):
    fig = plt.figure(figsize=(8, 6), facecolor="w", dpi=125)
    plt.imshow(ambiguity_surface, aspect="auto")
    print(ambiguity_surface.min(), ambiguity_surface.max())
    # _, im = plot_ambiguity_surface(
    #     ambiguity_surface,
    #     grid_data["rec_r"],
    #     grid_data["src_z"],
    #     imshow_kwargs={"vmin": 0, "vmax": 1},
    # )
    # plt.xlabel("Range [km]")
    # plt.ylabel("Depth [m]")
    # plt.title("MFP Ambiguity Surface")
    # cbar = plt.colorbar(im)
    # cbar.set_label("Normalized Correlation [dB]")
    return fig


def parse_args():
    parser = ArgumentParser(description="Plot MFP ambiguity surfaces.")
    parser.add_argument("source_path", type=Path, help="Source directory.")
    parser.add_argument(
        "--glob",
        default="*.npy",
        type=str,
        help="Glob pattern for ambiguity surface files.",
    )
    parser.add_argument(
        "--save_path", default=Path("plots"), type=Path, help="Plots directory."
    )
    parser.add_argument(
        "--grid_data",
        default=Path("grid_parameters.pkl"),
        type=Path,
        help="Path to grid data.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    files = get_filenames(args.source_path, args.glob)

    savepath = args.source_path / args.save_path
    savepath.mkdir(exist_ok=True, parents=True)

    grid_data = load_grid_data(args.source_path / args.grid_data)

    for f in files:
        data = load_ambiguity_surface(f)
        fig = plot(grid_data, abs(data))
        fig.savefig(savepath / f"{f.stem}.png", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()


# ../data/swellex96_S5_VLA/acoustic/ambiguity_surfaces/148-166-201-235-283-338-388_40x40
