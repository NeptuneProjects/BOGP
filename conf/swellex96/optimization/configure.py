#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path
import sys

from hydra.core.config_store import ConfigStore
from hydra_zen import builds

sys.path.insert(0, str(Path(__file__).parents[4]))
from data.swellex96.gps_range import load_range_csv
from optimization.parameterization import (
    IndexedParameterization,
    Parameterization,
    PermutedParameterization,
)
from conf.swellex96.optimization.common import SWELLEX96Paths
from conf.swellex96.optimization.run import LocalizationConf


# OBJECTIVE CONFIG
@dataclass
class ObjectiveConfig:
    name: str = "mfp"
    runner: str = "kraken"
    covariance_matrix: str = "simulation"
    parameters: str = "swellex96"
    beamformer: str = "bartlett"


ObjectiveConf = builds(
    ObjectiveConfig,
    name="mfp",
    runner="kraken",
    covariance_matrix="simulation",
    parameters="swellex96",
    beamformer="bartlett",
    builds_bases=(ObjectiveConfig,),
)


# METRIC CONFIG
@dataclass
class MetricConfig:
    name: str = "bartlett"
    minimize: bool = False


BartlettMetricConf = builds(
    MetricConfig,
    name="bartlett",
    minimize=False,
    builds_bases=(MetricConfig,),
)


# MONTE CARLO CONFIG
@dataclass
class MonteCarloConfig:
    seed: int
    num_mc_runs: int


SimulationMonteCarloConf = builds(
    MonteCarloConfig,
    seed=2009,
    num_mc_runs=100,
    builds_bases=(MonteCarloConfig,),
)
ExperimentalMonteCarloConf = builds(
    MonteCarloConfig,
    seed=2009,
    num_mc_runs=1,
    builds_bases=(MonteCarloConfig,),
)


# PARAMETERIZATION CONFIG
@dataclass
class ParameterizationConf:
    parameterization: Parameterization
    mode: str
    mc_config: MonteCarloConfig


SimulationParameterizationConf = builds(
    ParameterizationConf,
    parameterization=builds(
        Parameterization,
        permuted=builds(
            PermutedParameterization,
            scenario=builds(dict, rec_r=[1.0, 3.0, 5.0, 7.0]),
        ),
        fixed=builds(dict, src_z=60.0),
    ),
    mode="simulation",
    mc_config=SimulationMonteCarloConf,
    builds_bases=(ParameterizationConf,),
)

ExperimentalParameterizationConf = builds(
    ParameterizationConf,
    parameterization=builds(
        Parameterization,
        indexed=builds(
            IndexedParameterization,
            scenario=builds(
                dict,
                time_step=range(0, 350),
                rec_r=load_range_csv(SWELLEX96Paths.gps_data).tolist(),
            ),
        ),
        fixed=builds(dict, src_z=60.0, tilt=-1.0),
    ),
    mode="experimental",
    mc_config=ExperimentalMonteCarloConf,
    builds_bases=(ParameterizationConf,),
)


# PROBLEM TYPE CONFIG
@dataclass
class ProblemDefinition:
    name: str
    search_space: LocalizationConf


LocalizationDefinitionConf = builds(
    ProblemDefinition,
    name="localization",
    search_space=LocalizationConf,
    builds_bases=(ProblemDefinition,),
)


# CONFIG STORE
cs = ConfigStore.instance()
cs.store(
    group="parameterization", name="simulation", node=SimulationParameterizationConf
)
cs.store(
    group="parameterization",
    name="experimental",
    node=ExperimentalParameterizationConf,
)
cs.store(group="problem", name="localization", node=LocalizationDefinitionConf)
cs.store(group="objective", name="mfp", node=ObjectiveConf)
cs.store(group="metric", name="bartlett", node=BartlettMetricConf)
