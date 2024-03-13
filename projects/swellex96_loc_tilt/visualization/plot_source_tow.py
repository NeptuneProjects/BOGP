#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    DATADIR = Path(__file__).parents[4] / "data" / "swellex96_S5_VLA_loc_tilt" / "gps"
    df = pd.read_csv(DATADIR / "source_tow.csv")

    fig, axs = plt.subplots(nrows=5, figsize=(6, 12), sharex=True)

    ax = axs[0]
    ax.plot(df["Time"], df["Range [km]"], label="GPS Range")
    ax.plot(df["Time"], df["Apparent Range [km]"], label="Apparent Range")
    ax.legend(loc="upper right")
    ax.set_ylabel("Range [km]")
    
    ax = axs[1]
    ax.plot(df["Time"], df["Depth [m]"], label="Source Depth")
    ax.plot(df["Time"], df["Apparent Depth [m]"], label="Apparent Depth")
    ax.invert_yaxis()
    ax.legend(loc="right")
    ax.set_ylabel("Depth [m]")

    ax = axs[2]
    ax.plot(df["Time"], df["Bathy [m]"], label="Bathymetry")
    ax.invert_yaxis()
    ax.set_ylabel("Bathymetry [m]")
    
    ax = axs[3]
    ax.plot(df["Time"], df["Rel Az [deg]"], label="Rel. Azimuth")
    ax.set_ylabel("Rel. Azimuth [deg]")
    
    ax = axs[4]
    ax.plot(df["Time"], df["Apparent Tilt [deg]"], label="Apparent Tilt")
    ax.set_ylabel("Apparent Tilt [deg]")
    
    fig.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()
