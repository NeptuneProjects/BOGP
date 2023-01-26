#!/usr/bin/env python3

from datetime import datetime
from pathlib import Path

import configure
import process
import utils
from acoustics import aconfig

def main():
    cwd = Path.cwd()
    dt = datetime.now().strftime("%Y%m%d_%H%M%S")
    exppath = (cwd / "Data" / "Experiments" / "Results" / dt).relative_to(cwd)
    
    # Define experiment set
    # _ = configure.main(path=exppath)
    path_base = (cwd / "Data" / "Experiments" / "Localization2D" / "BaseEnvironments" / "SWELLEX96").relative_to(cwd)
    path_scene = (cwd / "Data" / "Experiments" / "Localization2D" / "Simulated" / "Scene001" / "ReceivedData").relative_to(cwd)
    aconfig.configure(path_base=path_base, path_scene=path_scene)
    
    # Load experiment configuration
    # exp = utils.Experiment()
    # exp.load_config(exppath / "config.json")
    # exp.create_folder_structure(root=exppath)

    # logger = utils.log(path=exppath)
    # logger.info("Commencing BOGP experiments.")
    # logger.info(f"Experiment run serial: {dt}")
    # logger.info("Experiment folder structure created.")
    # logger.info(f"{len(exp.experiments)} experiments in queue.")

    # Calculate received pressure field

    # Submit experiment configuration to worker function.
    # process.dispatcher(exp, n_workers=1)

    # logger.info("Complete.")
    

if __name__ == "__main__":
    main()
    