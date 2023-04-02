#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path
import time

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf

import numpy as np
from scipy.io import savemat
from tqdm import tqdm

from tritonoa.data import DataStream
from tritonoa.io.sio import SIODataHandler
from tritonoa.sp.processing import (
    FFTParameters,
    FrequencyParameters,
    FrequencyPeakFindingParameters,
    generate_complex_pressure,
)
from tritonoa.sp.timefreq import frequency_vector


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

    def __post_init__(self) -> None:
        self.paths.__post_init__()
        self.fvec = self.compute_fvec()

    def compute_fvec(self) -> np.ndarray:
        return frequency_vector(fs=self.fs, nfft=self.fft_params.nfft)


def convert(config: ConversionConfig):
    print("Converting SIO files to NUMPY files...")
    handler = SIODataHandler(files=config.paths.source.glob(config.glob_pattern))
    handler.convert_to_numpy(
        channels_to_remove=config.channels_to_remove,
        destination=config.paths.destination,
        max_workers=config.max_workers,
    )
    print("...conversion complete. ", end="")


def merge(config: MergeConfig):
    print("Merging numpy files...")
    handler = SIODataHandler(files=config.paths.source.glob(config.glob_pattern))
    handler.merge_numpy_files(
        base_time=config.base_time,
        start=config.start,
        end=config.end,
        fs=config.fs,
        channels_to_remove=config.channels_to_remove,
        savepath=config.paths.destination / config.filename,
    )
    print("...merge complete. ", end="")


def process(config: ProcessConfig):
    print(f"Processing data for {config.frequencies} Hz...")
    x, _ = DataStream.load(config.paths.source, exclude="t")
    print(f"Loaded data with shape: {x.shape}")

    x[:, 42] = x[:, [41, 43]].mean(axis=1)  # Remove corrupted channel
    x = np.fliplr(x)  # Reverse channel index

    for freq in tqdm(config.frequencies, bar_format="{l_bar}{bar:20}{r_bar}{bar:-20b}"):
        process_worker(
            data=x,
            num_segments=config.num_segments,
            freq_params=FrequencyParameters(
                freq=freq, fvec=config.fvec, peak_params=config.freq_finding_params
            ),
            fft_params=config.fft_params,
            destination=config.paths.destination,
        )
    print("...processing complete. ", end="")


def process_worker(data, num_segments, freq_params, fft_params, destination):
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


def run_operation(operation: str):
    return instantiate({"_target_": f"__main__.{operation}"}, _partial_=True)


OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
cs = ConfigStore.instance()
cs.store(name="conversion_config", node=ConversionConfig)
cs.store(name="merge_config", node=MergeConfig)
cs.store(name="process_config", node=ProcessConfig)


@hydra.main(
    config_path=str(Path(__file__).parents[2] / "conf" / "data" / "swellex96"),
    config_name="acoustics",
    version_base=None,
)
def main(cfg: ProcessConfig):
    print("Running acoustic processing pipeline.")
    start_pipeline = time.time()
    for command in cfg.run:
        start_command = time.time()
        config = instantiate(cfg.configs[command])
        run_operation(command)(config)
        print(f"{time.time() - start_command:.2f} s elapsed.")
    print(f"...pipeline complete. {time.time() - start_pipeline:.2f} s elapsed.")


if __name__ == "__main__":
    main()
