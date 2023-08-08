#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime
from itertools import product
from pathlib import Path
import sys
from typing import Optional, Type

import hydra
from hydra.core.config_store import ConfigStore
from hydra_zen import make_config, MISSING, save_as_yaml, instantiate
from hydra_zen.typing._implementations import DataClass
from oao.space import SearchSpaceBounds
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parents[1]))
from conf.common import FREQ, SWELLEX96Paths
import conf.configure as config_main
sys.path.insert(0, str(Path(__file__).parents[3]))
from optimization import utils


def configure(cfg: DictConfig) -> None:
    scenarios = instantiate(cfg.parameterization.parameterization).scenarios
    search_space = cfg.problem.search_space  # DO NOT INSTANTIATE!!!
    objective = instantiate(cfg.parameterization.objective)
    metric = instantiate(cfg.metric)
    mc = instantiate(cfg.parameterization.mc_config)

    main_path = cfg.path / cfg.problem.name / cfg.parameterization.mode / cfg.serial
    qpath = utils.make_serial_paths(main_path)
    seeds = utils.get_random_seeds(main_seed=mc.seed, num_mc_runs=mc.num_mc_runs)

    grid_search_configured = False
    for scenario, optimizer, seed in product(scenarios, cfg.optimizers, seeds):
        if not grid_search_configured and optimizer == "grid":
            seed = "0" * 9
            grid_search_configured = True

        savepath = utils.make_save_path(
            path=main_path, scenario=scenario, strategy=optimizer, seed=seed
        )

        Conf = configure_run(
            frequencies=cfg.frequencies,
            parameterization=scenario | {"tmpdir": savepath.parent.parent / "data"},
            optimizer=optimizer,
            objective=objective,
            search_space=search_space,
            seed=seed,
            monitor=cfg.monitor,
            formatter=cfg.formatter,
            metric_conf=metric,
            savepath=savepath,
        )
        save_as_yaml(
            Conf, qpath / utils.format_configfile_name(savepath, optimizer, seed)
        )


def configure_run(
    frequencies: list[int],
    parameterization: dict,
    optimizer: str,
    objective: config_main.ObjectiveConfig,
    search_space: SearchSpaceBounds,
    seed: int,
    formatter: str,
    metric_conf: config_main.MetricConfig,
    savepath: Path,
    monitor: Optional[callable] = None,
) -> Type[DataClass]:
    return make_config(
        defaults=[
            "_self_",
            {"optimizer": optimizer},
            {"objective": objective.name},
            {"objective/runner": objective.runner},
            {"objective/covariance_matrix": objective.covariance_matrix},
            {"objective/parameters": objective.parameters},
            {"objective/beamformer": objective.beamformer},
            {"formatter": formatter},
        ],
        frequencies=frequencies,
        parameterization=parameterization,
        optimizer=MISSING,
        objective=MISSING,
        search_space=search_space,
        seed=seed,
        monitor=monitor,
        formatter=MISSING,
        metric_name=metric_conf.name,
        minimize_objective=metric_conf.minimize,
        savepath=savepath,
    )


Config = make_config(
    defaults=[
        "_self_",
        {"parameterization": "simulation"},
        {"problem": "loc_tilt"},
        {"metric": "bartlett"},
    ],
    path=SWELLEX96Paths.outputs,
    serial=f"serial_{datetime.now().strftime('%Y%m%d%H%M%S')}",
    frequencies=FREQ,
    parameterization=MISSING,
    optimizers=["grid", "sobol", "gpei", "qgpei"],
    problem=MISSING,
    monitor=None,
    formatter="noiseless",
    metric=MISSING,
)


cs = ConfigStore.instance()
cs.store(name="config_main", node=Config)


@hydra.main(config_path=None, config_name="config_main", version_base="1.3")
def main(cfg: DictConfig) -> None:
    configure(cfg)


if __name__ == "__main__":
    main()
