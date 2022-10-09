#!/usr/bin/env python3
"""Script that aggregates results.
Example usage:
python3 ./Source/scripts/aggregate.py r l
"""
from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "Source"))
from BOGP import aggregate, utils

ROOT = Path.cwd() / "Data" / "Simulations"
# EXPERIMENT = ROOT / "range_estimation"
EXPERIMENT = ROOT / "localization"

EVALUATIONS = {
    "acq_func": ["ProbabilityOfImprovement", "ExpectedImprovement", "qExpectedImprovement"],
    # "acq_func": ["qExpectedImprovement"],
    "snr": ["inf", "20"],
    # "snr": ["inf"],
    "rec_r": ["0.5", "3.0", "6.0", "10.0"],
    # "rec_r": ["3.0", "6.0"],
    "src_z": ["62"]
}


def aggregate_runs():
    folders = utils.folders_of_evaluations(EVALUATIONS)
    for folder in folders:
        print(folder)
        Agg = aggregate.RangeEstimationAggregator(path=EXPERIMENT / folder)
        Agg.run(savepath=EXPERIMENT / folder / "aggregated.csv", verbose=True)


def aggregate_simulations():
    folders = utils.folders_of_evaluations(EVALUATIONS)
    df = utils.concatenate_simulation_results(
        [EXPERIMENT / folder / "aggregated.csv" for folder in folders]
    )
    df.to_csv(EXPERIMENT / "aggregated.csv")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        help="Select [r (runs)] or [s (simulations)]",
        choices=["r", "s"],
    )
    parser.add_argument(
        "simulation",
        type=str,
        choices=["r", "l"]
    )
    args = parser.parse_args()

    if args.mode == "r":
        aggregate_runs()
    elif args.mode == "s":
        aggregate_simulations()
