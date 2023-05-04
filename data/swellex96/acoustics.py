#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import time
from typing import Callable

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
from scipy.io import savemat
from tritonoa.data import DataStream
from tritonoa.io.sio import SIODataHandler
from tritonoa.sp.processing import (
    FFTParameters,
    FrequencyParameters,
    FrequencyPeakFindingParameters,
    generate_complex_pressure,
)
from tritonoa.sp.beamforming import covariance
from tritonoa.sp.timefreq import frequency_vector
from tqdm import tqdm

log = logging.getLogger(__name__)


@dataclass
class ProcessPaths:
    source: str
    destination: str

    def __post_init__(self):
        self.source = Path(self.source)
        self.destination = Path(self.destination)
        self.destination.mkdir(parents=True, exist_ok=True)


@dataclass
class ConversionConfig:
    paths: ProcessPaths
    glob_pattern: str
    channels_to_remove: list[int]
    max_workers: int

    def __post_init__(self):
        self.paths.__post_init__()


@dataclass
class CovarianceConfig:
    paths: ProcessPaths
    frequencies: list[float]
    num_segments: int

    def __post_init__(self):
        self.paths.__post_init__()


@dataclass
class MergeConfig:
    paths: ProcessPaths
    glob_pattern: str
    filename: str
    base_time: str
    start: str
    end: str
    fs: float
    channels_to_remove: list[int]

    def __post_init__(self):
        self.paths.__post_init__()


@dataclass
class ProcessConfig:
    paths: ProcessPaths
    frequencies: list[float]
    num_segments: int
    fs: float
    channels_to_remove: int
    freq_finding_params: FrequencyPeakFindingParameters
    fft_params: FFTParameters
    max_workers: int

    def __post_init__(self) -> None:
        self.paths.__post_init__()
        self.fvec = self.compute_fvec()

    def compute_fvec(self) -> np.ndarray:
        return frequency_vector(fs=self.fs, nfft=self.fft_params.nfft)



def compute_covariance(config: CovarianceConfig) -> None:
    log.info("Computing covariance matrices.")
    for freq in tqdm(
        config.frequencies,
        total=len(config.frequencies),
        desc="Processing frequencies",
        unit="freq",
        bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
    ):
        compute_covariance_worker(
            path=config.paths.source / f"{freq:.1f}Hz",
            timesteps=config.num_segments,
        )
    log.info("Covariance computation complete.")


def compute_covariance_worker(path: os.PathLike, timesteps: int):
    p = np.load(path / "data.npy")
    K = np.zeros((timesteps, p.shape[1], p.shape[1]), dtype=complex)
    for t in range(timesteps):
        d = np.expand_dims(p[t], 1)
        d /= np.linalg.norm(d)
        K[t] = covariance(d)

    np.save(path / "covariance.npy", K)


def convert(config: ConversionConfig) -> None:
    log.info("Converting SIO files to NUMPY files.")
    handler = SIODataHandler(files=config.paths.source.glob(config.glob_pattern))
    handler.convert_to_numpy(
        channels_to_remove=config.channels_to_remove,
        destination=config.paths.destination,
        max_workers=config.max_workers,
    )
    log.info("Conversion complete.")


def merge(config: MergeConfig) -> None:
    log.info("Merging numpy files.")
    handler = SIODataHandler(files=config.paths.source.glob(config.glob_pattern))
    handler.merge_numpy_files(
        base_time=config.base_time,
        start=config.start,
        end=config.end,
        fs=config.fs,
        channels_to_remove=config.channels_to_remove,
        savepath=config.paths.destination / config.filename,
    )
    log.info("Merge complete.")


def process(config: ProcessConfig) -> None:
    log.info(f"Processing data for {config.frequencies} Hz.")
    x, _ = DataStream.load(config.paths.source, exclude="t")
    log.info(f"Loaded data with shape: {x.shape}")

    x[:, 42] = x[:, [41, 43]].mean(axis=1)  # Remove corrupted channel
    # x -= x.mean(axis=0) # Remove mean
    x = np.fliplr(x)  # Reverse channel index

    with tqdm(
        total=len(config.frequencies),
        desc="Processing frequencies",
        unit="freq",
        bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}",
    ) as pbar:
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = [
                executor.submit(
                    process_worker,
                    data=x,
                    num_segments=config.num_segments,
                    freq_params=FrequencyParameters(
                        freq=freq,
                        fvec=config.fvec,
                        peak_params=config.freq_finding_params,
                    ),
                    fft_params=config.fft_params,
                    destination=config.paths.destination,
                )
                for freq in config.frequencies
            ]
            [pbar.update(1) for _ in as_completed(futures)]

    log.info("Processing complete.")


def process_worker(
    data: np.ndarray,
    num_segments: int,
    freq_params: FrequencyParameters,
    fft_params: FFTParameters,
    destination: os.PathLike,
) -> None:
    def _save_data():
        savepath = destination / f"{freq_params.freq:.1f}Hz"
        savepath.mkdir(parents=True, exist_ok=True)
        np.save(savepath / "data.npy", p)
        np.save(savepath / "f_hist.npy", f_hist)
        savemat(
            savepath / f"data_{freq_params.freq}Hz.mat", {"p": p, "f": freq_params.freq}
        )

    p, f_hist = generate_complex_pressure(
        data=data,
        num_segments=num_segments,
        freq_params=freq_params,
        fft_params=fft_params,
    )
    _save_data()


def run_operation(operation: str) -> Callable:
    return instantiate({"_target_": f"__main__.{operation}"}, _partial_=True)


OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
cs = ConfigStore.instance()
cs.store(name="conversion_config", node=ConversionConfig)
cs.store(name="merge_config", node=MergeConfig)
cs.store(name="process_config", node=ProcessConfig)


@hydra.main(
    config_path=str(Path(__file__).parents[3] / "conf" / "swellex96" / "data"),
    config_name="acoustics",
    version_base=None,
)
def main(cfg: ProcessConfig):
    log.info("Running acoustic processing pipeline.")
    start_pipeline = time.time()
    for command in cfg.run:
        start_command = time.time()
        config = instantiate(cfg.configs[command])
        run_operation(command)(config)
        log.info(f"{time.time() - start_command:.2f} s elapsed.")
    log.info(f"Pipeline complete. {time.time() - start_pipeline:.2f} s elapsed.")


if __name__ == "__main__":
    main()
