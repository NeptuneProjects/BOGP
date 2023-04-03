#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import sys

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from tritonoa.at.models.kraken.kraken import clean_up_kraken_files
from tritonoa.at.models.kraken.runner import run_kraken as runner
from tritonoa.sp.beamforming import beamformer
from tritonoa.sp.mfp import MatchedFieldProcessor
sys.path.insert(0, str(Path(__file__).parents[2]))
from BOGP import parameterization as param
from env import load_env_from_json
from gps_range import load_range_csv

log = logging.getLogger(__name__)


@dataclass
class MFPPaths:
    data_path: str
    env_path: str
    range_path: str
    mfp_path: str
    covariance_path: str

    def __post_init__(self):
        self.data_path = Path(self.data_path)
        self.env_path = Path(self.env_path)
        self.range_path = Path(self.range_path)
        self.mfp_path = Path(self.mfp_path)
        self.covariance_path = Path(self.covariance_path)


@dataclass
class RunConfig:
    paths: MFPPaths
    frequencies: list[float]
    num_segments: int
    num_elements: int
    max_workers: int
    parameters: dict
    mfp_parameters: dict

    def __post_init__(self):
        self.paths.__post_init__()


def format_dict_config(dict_config: DictConfig) -> dict:
    if dict_config is None:
        return {}
    return OmegaConf.to_object(dict_config)


def initialize_fixed_param(env_path: os.PathLike, fixed_def: dict) -> dict:
    environment = load_env_from_json(env_path)
    return environment | fixed_def


def initialize_indexed_param(
    range_path: os.PathLike, indexed_def: dict
) -> param.IndexedParameterization:
    ranges = load_range_csv(range_path)
    return param.IndexedParameterization(
        scenario=indexed_def | {"rec_r": ranges},
    )


def initialize_permuted_param(permuted_def: dict) -> param.PermutedParameterization:
    return param.PermutedParameterization(
        scenario=permuted_def,
    )


def initialize_scenarios(cfg: RunConfig):
    indexed_def = format_dict_config(cfg.parameters["indexed"])
    fixed_def = format_dict_config(cfg.parameters["fixed"])
    permuted_def = format_dict_config(cfg.parameters["permuted"])

    indexed_param = initialize_indexed_param(
        range_path=cfg.paths.range_path,
        indexed_def=indexed_def,
    )
    permuted_param = initialize_permuted_param(
        permuted_def=permuted_def,
    )
    fixed_param = initialize_fixed_param(
        env_path=cfg.paths.env_path,
        fixed_def=fixed_def,
    )
    return param.Parameterization(
        indexed=indexed_param,
        permuted=permuted_param,
        fixed=fixed_param,
    )


def load_covariance(path, frequencies, num_segments, num_elements):
    """Returns array of covariance matrices:
    (num_frequencies, num_segments, num_elements, num_elements)
    """
    K = np.zeros(
        (len(frequencies), num_segments, num_elements, num_elements), dtype=complex
    )
    for i, f in enumerate(frequencies):
        K[i] = np.load(path / f"{f:.1f}Hz" / "covariance.npy")

    return K


def run_parameterizations(cfg, parameterization: param.Parameterization):
    log.info("Generating ambiguity surfaces for each scenario.")
    savepath = cfg.paths.mfp_path / "".join([f"{f}-" for f in cfg.frequencies])[:-1]
    log.info("Loading covariance matrices.")
    K = load_covariance(
        path=cfg.paths.covariance_path,
        frequencies=cfg.frequencies,
        num_segments=cfg.num_segments,
        num_elements=cfg.num_elements,
    )

    with tqdm(
        total=len(parameterization),
        desc="Creating ambiguity surfaces",
        unit="scenario",
        bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
    ) as pbar:
        with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
            futures = [
                executor.submit(
                    worker,
                    scenario,
                    savepath,
                    cfg,
                    K[:, i, ...],
                )
                for i, scenario in enumerate(parameterization)
            ]
            [pbar.update(1) for _ in as_completed(futures)]

    log.info("Processing complete.")


def worker(
    scenario: dict,
    savepath: os.PathLike,
    cfg: RunConfig,
    K: np.ndarray,
):
    def _save_results():
        np.save(savepath / f"surface_{scenario['title']}.npy", amb_surface)
        # np.save(savepath / f"covariance_{scenario['title']}.npy", K)

    scenario["tmpdir"] = savepath / scenario["tmpdir"]
    scenario["title"] = f"{scenario['timestep']:03d}"
    scenario["tmpdir"].mkdir(parents=True, exist_ok=True)

    MFP = MatchedFieldProcessor(
        runner=runner,
        covariance_matrix=K,
        freq=cfg.frequencies,
        parameters=scenario,
        beamformer=beamformer,
    )

    amb_surface = np.zeros(
        (
            len(cfg.mfp_parameters.search_space.src_z),
            len(cfg.mfp_parameters.search_space.rec_r),
        )
    )
    for zz, z in enumerate(cfg.mfp_parameters.search_space.src_z):
        amb_surface[zz] = MFP.evaluate(
            {"src_z": z, "src_r": cfg.mfp_parameters.search_space.rec_r}
        )

    clean_up_kraken_files(scenario["tmpdir"])
    _save_results()


OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
cs = ConfigStore.instance()
cs.store(name="run_config", node=RunConfig)


@hydra.main(
    config_path=str(Path(__file__).parents[2] / "conf" / "data" / "swellex96"),
    config_name="mfp",
    version_base=None,
)
def main(_cfg: RunConfig):
    cfg = instantiate(_cfg, _recursive_=True)
    parameterization = initialize_scenarios(cfg)
    run_parameterizations(cfg, parameterization)


if __name__ == "__main__":
    main()
