#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
import sys

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from hydra_zen import make_config, MISSING
from oao.results import get_results
from omegaconf import DictConfig, OmegaConf
from tritonoa.at.models.kraken.kraken import clean_up_kraken_files

sys.path.insert(0, str(Path(__file__).parents[1]))
import conf.run as config_run
sys.path.insert(0, str(Path(__file__).parents[3]))
from optimization import utils


def task_fn(cfg: DictConfig) -> None:
    parameterization = OmegaConf.to_object(cfg.parameterization)
    objective = config_run.format_objective(cfg, config_run.instantiate_objective(cfg))
    search_space = utils.format_search_space(
        instantiate(cfg.search_space), parameterization
    )
    strategy = instantiate(cfg.optimizer.strategy)(seed=cfg.seed)

    optimizer = instantiate(cfg.optimizer)(
        objective=objective,
        search_space=search_space,
        strategy=strategy(),
    )
    optimizer.run(name="demo_grid")

    get_results(
        client=optimizer.client,
        times=optimizer.batch_execution_times,
        minimize=objective.properties.minimize,
    ).to_csv(cfg.savepath / "results.csv")
    optimizer.client.save_to_json_file(cfg.savepath / "client.json")
    try:
        utils.move_config_to_savepath(cfg.savepath)
    except FileNotFoundError as e:
        logging.warning(f"{e}")
    finally:
        OmegaConf.save(cfg, cfg.savepath / "configured.yaml")
    clean_up_kraken_files(cfg.savepath)


Config = make_config(
    defaults=[
        "_self_",
        {"optimizer": "sobol"},
        {"objective": "mfp"},
        {"objective/runner": "kraken"},
        {"objective/covariance_matrix": "simulation"},
        {"objective/parameters": "swellex96"},
        {"objective/beamformer": "bartlett_ml"},
        {"objective/multifreq_method": "product"},
        {"formatter": "noiseless"},
    ],
    frequencies=[201.0, 235.0, 283.0, 338.0, 388.0],
    parameterization={"rec_r": 1.0, "src_z": 60.0, "time_step": 250},
    optimizer=MISSING,
    objective=MISSING,
    search_space=config_run.SearchConf,
    seed=2009,
    monitor=None,
    formatter=MISSING,
    metric_name="bartlett",
    minimize_objective=False,
    savepath=Path.cwd(),
)

cs = ConfigStore.instance()
cs.store(name="config_run", node=Config)


@hydra.main(config_path=None, config_name="config_run", version_base="1.3")
def main(cfg: DictConfig) -> None:
    return task_fn(cfg)


if __name__ == "__main__":
    main()
