from argparse import ArgumentParser
from pathlib import Path
import sys
from typing import Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from tritonoa.plotting import plot_ambiguity_surface

sys.path.insert(0, str(Path(__file__).parents[1]))
from conf import common

sys.path.insert(0, str(Path(__file__).parents[3]))
from optimization import utils

GPS_DATA = pd.read_csv(common.SWELLEX96Paths.gps_data)


def load_data(source: Path, glob_pattern: str) -> list[np.ndarray]:
    return [(np.load(f), f.stem) for f in Path(source).glob(glob_pattern)]


def get_gps_data(step: str) -> pd.DataFrame:
    df = GPS_DATA.iloc[int(step)]

    r_min = 0.5
    r_max = 8.0

    r = df["Apparent Range [km]"]

    rl_500, ru_500 = utils.adjust_bounds(r - 0.5, r_min, r + 0.5, r_max)
    rl_1000, ru_1000 = utils.adjust_bounds(r - 1.0, r_min, r + 1.0, r_max)
    zl_20 = 40.0
    zu_20 = 80.0
    zl_40 = 20.0
    zu_40 = 100.0

    rect_small = [(rl_500, zl_20), 1.0, 40.0]
    rect_big = [(rl_1000, zl_40), 2.0, 80.0]

    return rect_small, rect_big

def create_figure(
    data: np.ndarray,
    grid: dict[np.ndarray],
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
    minimize: bool = True,
    step: Optional[str] = None,
) -> plt.Figure:
    if minimize:
        data = 1.0 - data
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()


    small_box, big_box = get_gps_data(step)
    
    fig = plt.figure(figsize=(8, 6))
    ax, im = plot_ambiguity_surface(
        data,
        rvec=grid["rec_r"],
        zvec=grid["src_z"],
        imshow_kwargs={"vmin": vmin, "vmax": vmax, "cmap": cmap},
    )
    rect = patches.Rectangle(*small_box, zorder=100, facecolor="none", edgecolor="red", linewidth=2)
    ax.add_patch(rect)
    rect = patches.Rectangle(*big_box, zorder=100, facecolor="none", edgecolor="orange", linewidth=2)
    ax.add_patch(rect)

    ax.set_xlabel("Range [km]")
    ax.set_ylabel("Depth [m]")
    ax.set_title(f"Time Step {step}")
    fig.colorbar(im, ax=ax)
    return fig


def save_figure(destination: Path, fig: plt.Figure) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=300)


def main(args: ArgumentParser) -> None:
    grid = np.load(args.source / "grid_parameters.pkl", allow_pickle=True)
    data = load_data(args.source, args.glob)
    pbar = tqdm(
        data,
        total=len(data),
        desc="Creating figures",
        unit="figure",
        bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
    )
    for surf, fname in pbar:
        fig = create_figure(
            surf,
            grid,
            vmin=args.vmin,
            vmax=args.vmax,
            cmap=args.cmap,
            minimize=args.minimize,
            step=fname.split('_')[-1],
        )
        save_figure(args.source / args.destination / (fname + "." + args.fmt), fig)
        plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--source",
        "--s",
        type=Path,
        default="../data/swellex96_S5_VLA_inv/acoustic/processed_000/ambiguity_surfaces/49-64-79-94-112-130-148-166-201-235-283-338-388_100x40",
    )
    parser.add_argument("--glob", "--g", type=str, default="surface_*.npy")
    parser.add_argument("--destination", "--d", type=str, default="plots/rel_scale")
    parser.add_argument("--fmt", "--f", type=str, default="png")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--cmap", type=str, default="jet")
    parser.add_argument("--minimize", default=False, action="store_true")
    args = parser.parse_args()
    main(args)
