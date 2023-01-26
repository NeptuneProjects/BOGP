#!/usr/bin/env python3

"""This scripts reads in acoustic data files from SWELLEX96 event S-5,
concatenates the data, removes data extraneous to the analysis, and saves
the merged data to file.
"""

import datetime
from pathlib import Path

import numpy as np

DATADIR = Path.cwd().parent / "DATA" / "SWELLEX96" / "VLA" / "selected"
FS = 1500  # Sampling rate [Hz]


def main():
    # Load data files
    files = sorted([f for f in DATADIR.glob("*.npz")])
    data = []
    for f in files:
        data.append(np.load(f)["X"])

    # Remove extraneous channel
    data = np.concatenate(data)[:, 0:-1]
    # Define time vector [s]
    t = np.linspace(0, (data.shape[0] - 1) / FS, data.shape[0])
    # Specify starting file datetime
    base_time = datetime.datetime.strptime("96131 23:02", "%y%j %H:%M")
    # Specify analysis starting datetime
    start = datetime.datetime.strptime("96131 23:19", "%y%j %H:%M")
    # Specify analysis ending datetime
    end = datetime.datetime.strptime("96132 00:22", "%y%j %H:%M")
    # Create datetime vector
    dt = np.array([base_time + datetime.timedelta(seconds=i) for i in t])
    # Find indeces of analysis data
    idx = (dt >= start) & (dt < end)
    # Remove extraneous data
    data = data[idx]
    t = t[idx]
    # Save to file
    np.savez(DATADIR / "merged", X=data, t=t)


if __name__ == "__main__":
    main()
