#!/usr/bin/env python3

import csv
import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def load_range_csv(path):
    return np.loadtxt(path, delimiter=",", skiprows=1, usecols=2)


def save_range_csv(path, data, header):
    DATADIR = Path(
        "/Users/williamjenkins/Research/Projects/BOGP/Data/SWELLEX96/VLA/selected"
    )
    BASE_TIME = datetime.datetime.strptime("96131 23:02", "%y%j %H:%M")

    start = pd.to_datetime(pd.Timestamp(1996, 5, 10, 23, 21, 0))
    end = pd.to_datetime(pd.Timestamp(1996, 5, 11, 0, 24, 0))

    t = pd.date_range(start, end=end, periods=350)

    df = pd.read_fwf(DATADIR.parent.parent / "RangeEventS5" / "SproulToVLA.S5.txt")
    df["Datetime"] = pd.to_datetime(
        "96" + df["Jday"].astype(str) + " " + df["Time"], format="%y%j %H:%M"
    )
    df = df.drop(columns=["Duration", "Jday", "Time"]).set_index("Datetime")
    df = df.loc[start:end]
    r = df["Range(km)"].values

    r_ts = np.interp(t, df.index, r)

    df2 = pd.DataFrame({"Time": t, "Range [km]": r_ts})
    df2.to_csv(DATADIR / "gps_range.csv")




def main():
    data = load_range_csv("/Users/williamjenkins/Research/Projects/BOGP/data/swellex96_S5_VLA/gps/gps_range.csv")
    print(data)
    print(data.shape)



if __name__ == "__main__":
    main()
