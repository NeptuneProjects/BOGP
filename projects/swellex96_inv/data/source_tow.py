#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

import bathyreq
import numpy as np
import pandas as pd
from pyproj import Geod
from tritonoa.at.env.array import compute_range_offsets

sys.path.insert(0, str(Path(__file__).parents[1]))
from conf.common import SWELLEX96Paths
from env import load_from_json

DECLINATION: float = 13.35  # From IGRF model hosted by NOAA NCEI
NUM_SEGMENTS: int = 123
TRUE_ARRAY_DEPTH: float = 216.0
TRUE_DEPTH: float = 60.0

# Define start/end times for resampling
start = pd.to_datetime(pd.Timestamp(1996, 5, 10, 23, 30, 0))
end = pd.to_datetime(pd.Timestamp(1996, 5, 11, 0, 15, 0))
t = pd.date_range(start, end=end, periods=NUM_SEGMENTS)


def load_tilt_data(path: Path) -> pd.DataFrame:
    """Load tilt data from file."""
    df = pd.read_fwf(
        path, names=["Date", "Time", "Depth", "Temp", "Tilt [deg]", "Azimuth"]
    )
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df["Azimuth [deg]"] = df["Azimuth"] + DECLINATION
    df = df.drop(columns=["Date", "Time", "Depth", "Temp", "Azimuth"])
    df = df.set_index("Datetime")
    return df


def add_bathymetry(df: pd.DataFrame) -> pd.DataFrame:
    """Add bathymetry data to GPS data.

    Parameters
    ----------
    df : pd.DataFrame
        GPS data.

    Returns
    -------
    pd.DataFrame
        GPS data with bathymetry data.
    """
    req = bathyreq.BathyRequest()
    df["Bathy [m]"] = -req.get_point(
        longitude=df["Lon"].values, latitude=df["Lat"].values, size=[400, 400]
    )
    return df


def add_tilt_data(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Add array tilt data columns to GPS data.

    Parameters
    ----------
    df : pd.DataFrame
        GPS data.

    Returns
    -------
    pd.DataFrame
        GPS data with array tilt data.
    """
    df_tilt = resample_tilt_df(load_tilt_data(path))
    new_df = pd.merge(df, df_tilt, how="inner")
    new_df["Rel Az [deg]"] = new_df["Azimuth [deg]"] - new_df["Src Brg [deg]"]
    return new_df


def compute_apparent_tilt(df: pd.DataFrame) -> pd.DataFrame:
    """Compute apparent tilt in the LOS from GPS data.

    Parameters
    ----------
    df : pd.DataFrame
        GPS data.

    Returns
    -------
    pd.DataFrame
        GPS data with apparent tilt added.
    """
    rec_z = np.array(load_from_json(SWELLEX96Paths.environment_data)["rec_z"])
    scope = TRUE_ARRAY_DEPTH - rec_z.min()
    rec_r = df["Apparent Range [km]"].values[0]
    tilt = df["Tilt [deg]"].values[0]
    azimuth = df["Rel Az [deg]"].values[0]

    apparent_tilt = []
    for i in range(len(df)):
        rec_r = df["Apparent Range [km]"].values[i]
        tilt = df["Tilt [deg]"].values[i]
        azimuth = df["Rel Az [deg]"].values[i]
        z_pivot = TRUE_ARRAY_DEPTH
        range_offsets, _ = compute_range_offsets(rec_r, rec_z, tilt, azimuth, z_pivot)
        apparent_tilt.append(
            np.rad2deg(
                np.arcsin(-range_offsets[np.argmax(np.abs(range_offsets))] / scope)
            )
        )

    df["Apparent Tilt [deg]"] = apparent_tilt
    return df


def compute_mirage(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mirage parameters from GPS data.

    Parameters
    ----------
    df : pd.DataFrame
        GPS data.

    Returns
    -------
    pd.DataFrame
        GPS data with mirage parameters.
    """
    df["Apparent Range [km]"] = TRUE_ARRAY_DEPTH * df["Range [km]"] / df["Bathy [m]"]
    df["Apparent Depth [m]"] = TRUE_ARRAY_DEPTH * df["Depth [m]"] / df["Bathy [m]"]
    return df


def compute_range_and_bearing_to_source(df: pd.DataFrame) -> pd.DataFrame:
    """Compute range and bearing from VLA to source using GPS data.

    VLA position: 32°40.254'N, 117°21.620'W

    Parameters
    ----------
    df : pd.DataFrame
        GPS data.

    Returns
    -------
    pd.DataFrame
        GPS data with range and bearing to source.
    """

    VLA_POSIT = (len(df) * [32 + 40.254 / 60], len(df) * [-117 - 21.620 / 60])
    g = Geod(ellps="WGS84")
    _, az2, r = g.inv(df["Lon"].values, df["Lat"].values, VLA_POSIT[1], VLA_POSIT[0])
    df["Range [km]"] = np.array(r) / 1000
    az2[az2 < 0] = az2[az2 < 0] + 360
    df["Src Bearing [deg]"] = np.array(az2)
    return df


def format_gps_data(df: pd.DataFrame) -> pd.DataFrame:
    """Format GPS data to decimal degrees.

    Parameters
    ----------
    df : pd.DataFrame
        GPS data.

    Returns
    -------
    pd.DataFrame
        GPS data with decimal degrees.
    """
    df["Lon"] = -(df["Lon [Deg]"].astype(float) + df["Lon [Min]"].astype(float) / 60)
    df["Lat"] = df["Lat [Deg]"].astype(float) + df["Lat [Min]"].astype(float) / 60
    return df.drop(columns=["Lat [Deg]", "Lat [Min]", "Lon [Deg]", "Lon [Min]"])


def format_gps_time(df: pd.DataFrame) -> pd.DataFrame:
    """Format GPS time to datetime. GPS device was 1 m 3 s behind true time.

    Parameters
    ----------
    df : pd.DataFrame
        GPS data.

    Returns
    -------
    pd.DataFrame
        GPS data with datetime index.
    """
    df["Datetime"] = pd.to_datetime(
        "96" + df["Jday"] + " " + df["Time [H]"] + ":" + df["Time [M]"],
        format="%y%j %H:%M",
    ) + pd.Timedelta(minutes=1, seconds=3)
    return df.drop(columns=["Jday", "Time [H]", "Time [M]"]).set_index("Datetime")


def load_gps_data(fname: Path) -> pd.DataFrame:
    """Load GPS data from file.

    Parameters
    ----------
    fname : Path
        Path to GPS data file.

    Returns
    -------
    pd.DataFrame
        Unformatted GPS data.
    """
    return pd.read_fwf(
        fname,
        names=[
            "Jday",
            "Time [H]",
            "Time [M]",
            "Lat [Deg]",
            "Lat [Min]",
            "Lon [Deg]",
            "Lon [Min]",
        ],
        dtype=str,
    )


def resample_main_df(df: pd.DataFrame) -> pd.DataFrame:
    """Resamples dataframe according to desired timesteps between start/end times.

    Parameters
    ----------
    df : pd.DataFrame
        GPS data.

    Returns
    -------
    pd.DataFrame
        Resampled GPS data.
    """

    # t = pd.date_range(start, end=end, periods=250)

    # Trim and extract data
    df = df.loc[start:end]
    lon = df["Lon"].values
    lat = df["Lat"].values
    r = df["Range [km]"].values
    brg = df["Src Bearing [deg]"].values

    # Interpolate data to new timebase
    lon_ts = np.interp(t, df.index, lon)
    lat_ts = np.interp(t, df.index, lat)
    r_ts = np.interp(t, df.index, r)
    brg_ts = np.interp(t, df.index, brg)

    return pd.DataFrame(
        {
            "Time": t,
            "Lon": lon_ts,
            "Lat": lat_ts,
            "Range [km]": r_ts,
            "Src Brg [deg]": brg_ts,
        }
    )


def resample_tilt_df(df: pd.DataFrame) -> pd.DataFrame:
    """Resamples dataframe according to desired timesteps between start/end times.

    Parameters
    ----------
    df : pd.DataFrame
        GPS data.

    Returns
    -------
    pd.DataFrame
        Resampled GPS data.
    """

    # Trim and extract data
    # start_local = start - pd.to_timedelta(7, unit="h")
    # end_local = end - pd.to_timedelta(7, unit="h")
    # df = df.loc[start_local:end_local]
    df = df.loc[start:end]
    azimuth = df["Azimuth [deg]"].values
    tilt = df["Tilt [deg]"].values

    # Interpolate data to new timebase
    azimuth_ts = np.interp(t, df.index, azimuth)
    tilt_ts = np.interp(t, df.index, tilt)

    return pd.DataFrame(
        {
            "Time": t,
            "Azimuth [deg]": azimuth_ts,
            "Tilt [deg]": tilt_ts,
        }
    )


def main() -> None:
    DATADIR = Path(__file__).parents[4] / "data" / "swellex96_S5_VLA_inv" / "gps"
    df = load_gps_data(DATADIR / "EventS5.txt")
    df = format_gps_time(df)
    df = format_gps_data(df)
    df = compute_range_and_bearing_to_source(df)
    df = resample_main_df(df)
    df["Depth [m]"] = TRUE_DEPTH
    df = add_bathymetry(df)
    df = compute_mirage(df)
    df = add_tilt_data(df, DATADIR / "SW96_1.dat")
    df = compute_apparent_tilt(df)
    df.to_csv(DATADIR / "source_tow.csv")


if __name__ == "__main__":
    main()
