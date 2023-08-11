from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from tritonoa.plotting import plot_ambiguity_surface


def load_data(source: Path, glob_pattern: str) -> list[np.ndarray]:
    return [(np.load(f), f.stem) for f in Path(source).glob(glob_pattern)]


def create_figure(
    data: np.ndarray,
    grid: dict[np.ndarray],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
) -> plt.Figure:
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    fig = plt.figure(figsize=(8, 6))
    ax, im = plot_ambiguity_surface(
        data,
        rvec=grid["rec_r"],
        zvec=grid["src_z"],
        imshow_kwargs={"vmin": vmin, "vmax": vmax, "cmap": cmap},
    )
    ax.set_xlabel("Range [km]")
    ax.set_ylabel("Depth [m]")
    fig.colorbar(im, ax=ax)
    return fig


def save_figure(destination: Path, fig: plt.Figure) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=300)


def main(args: ArgumentParser) -> None:
    grid = np.load(args.source / "grid_parameters.pkl", allow_pickle=True)
    data = load_data(args.source, args.glob)
    for surf, fname in data:
        fig = create_figure(surf, grid, vmin=args.vmin, vmax=args.vmax, cmap=args.cmap)
        save_figure(args.source / args.destination / (fname + "." + args.fmt), fig)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--source",
        "--s",
        type=Path,
        default="/Users/williamjenkins/Research/Projects/BOGP/data/swellex96_S5_VLA_loc_tilt/acoustic/ambiguity_surfaces/201-235-283-338-388_200x100",
    )
    parser.add_argument("--glob", "--g", type=str, default="surface_*.npy")
    parser.add_argument("--destination", "--d", type=str, default="plots/rel_scale")
    parser.add_argument("--fmt", "--f", type=str, default="png")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--cmap", type=str, default="viridis")
    args = parser.parse_args()
    main(args)
