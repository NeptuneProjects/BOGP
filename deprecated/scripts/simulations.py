#!/usr/bin/env python3

"""Script containing simulations for paper.
Example usage:
python3 ./Source/scripts/simulations.py r Source/config/config.json --path ./Data/Simulations --workers 10
python3 ./Source/scripts/simulations.py r Source/config/config.json --path ./Data/Simulations --workers 4 --device cuda
"""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path.cwd() / "Source"))
from BOGP import production


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify which simulations to run.")
    parser.add_argument(
        "simulation",
        type=str,
        help="Simulation name.",
        choices=["range", "r", "localize", "l"],
    )
    parser.add_argument("configpath", type=str, help="Path to configuration file.")
    parser.add_argument("--workers", type=int, help="Number of workers.", default=1)
    parser.add_argument(
        "--path", type=str, help="Path to save results.", default=Path.cwd()
    )
    parser.add_argument("--device", type=str, help="cpu or cuda", default="cpu")
    args = parser.parse_args()

    kwargs = {
        "workers": args.workers,
        "path": Path(args.path) if isinstance(args.path, str) else args.path,
        "device": args.device
    }
    with open(Path(args.configpath), "r") as fp:
        config = json.load(fp)
    config["configpath"] = args.configpath

    simulator = production.Simulator(config=config)
    simulator.simulate(args.simulation, **kwargs)
    
