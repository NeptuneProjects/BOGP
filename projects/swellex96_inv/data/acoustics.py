#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Callable

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
from tritonoa.data import DataStream
from tritonoa.io.sio import SIODataHandler
from tritonoa.sp.processing import (
    FFTParameters,
    FrequencyPeakFindingParameters,
    Processor,
)
from tritonoa.sp.timefreq import frequency_vector

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
    fmt: list[str]
    paths: ProcessPaths
    glob_pattern: str
    channels_to_remove: list[int]
    max_workers: int
    fs: int

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
    samples_per_segment: int
    segments_every_n: int
    fs: float
    channels_to_remove: int
    reverse_channels: bool
    freq_finding_params: FrequencyPeakFindingParameters
    fft_params: FFTParameters
    max_workers: int
    normalize_covariance: bool
    covariance_averaging: int

    def __post_init__(self) -> None:
        self.paths.__post_init__()
        self.fvec = self.compute_fvec()

    def compute_fvec(self) -> np.ndarray:
        return frequency_vector(fs=self.fs, nfft=self.fft_params.nfft)


def convert(config: ConversionConfig) -> None:
    log.info("Converting SIO files to specified formats.")
    handler = SIODataHandler(files=config.paths.source.glob(config.glob_pattern))
    handler.convert(
        fmt=config.fmt,
        channels_to_remove=config.channels_to_remove,
        destination=config.paths.destination,
        max_workers=config.max_workers,
        fs=config.fs,
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
    ds = DataStream()
    ds.load(config.paths.source, exclude="t")
    log.info(f"Loaded data with shape: {ds.X.shape}")
    if config.channels_to_remove is not None:
        for channel in config.channels_to_remove:
            ds.X[:, channel] = 0.0
    if config.reverse_channels:
        ds.X = np.fliplr(ds.X)

    processor = Processor(
        data=ds,
        fs=config.fs,
        freq=config.frequencies,
        fft_params=config.fft_params,
        freq_finding_params=config.freq_finding_params,
    )
    processor.process(
        samples_per_segment=config.samples_per_segment,
        segments_every_n=config.segments_every_n,
        destination=config.paths.destination,
        max_workers=config.max_workers,
        normalize_covariance=config.normalize_covariance,
        covariance_averaging=config.covariance_averaging,
    )


def run_operation(operation: str) -> Callable:
    return instantiate({"_target_": f"__main__.{operation}"}, _partial_=True)


OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
cs = ConfigStore.instance()
cs.store(name="conversion_config", node=ConversionConfig)
cs.store(name="merge_config", node=MergeConfig)
cs.store(name="process_config", node=ProcessConfig)


@hydra.main(
    config_path=str(Path(__file__).parents[1] / "conf"),
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
