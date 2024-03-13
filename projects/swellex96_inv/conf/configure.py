#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path
import sys

from hydra.core.config_store import ConfigStore
from hydra_zen import builds
import pandas as pd

from conf.common import SWELLEX96Paths
from conf.run import SearchConf

sys.path.insert(0, str(Path(__file__).parents[3]))
from optimization.parameterization import (
    IndexedParameterization,
    Parameterization,
)

START = 0
END = 125

# OBJECTIVE CONFIG
@dataclass
class ObjectiveConfig:
    name: str = "mfp"
    runner: str = "kraken"
    covariance_matrix: str = "simulation"
    parameters: str = "swellex96"
    beamformer: str = "bartlett"


ExperimentalObjectiveConf = builds(
    ObjectiveConfig,
    name="mfp",
    runner="kraken",
    covariance_matrix="experimental",
    parameters="swellex96",
    beamformer="bartlett_ml",
    builds_bases=(ObjectiveConfig,),
)


SimulationObjectiveConf = builds(
    ObjectiveConfig,
    name="mfp",
    runner="kraken",
    covariance_matrix="simulation",
    parameters="swellex96",
    beamformer="bartlett_ml",
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


BartlettMLMetricConf = builds(
    MetricConfig,
    name="bartlett_ml",
    minimize=True,
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
    num_mc_runs=1,
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
    objective: ObjectiveConfig
    mc_config: MonteCarloConfig


SimulationParameterizationConf = builds(
    ParameterizationConf,
    parameterization=builds(
        Parameterization,
        indexed=builds(
            IndexedParameterization,
            scenario=builds(
                dict,
                time_step=range(0, 250)[START:END],
                rec_r=pd.read_csv(SWELLEX96Paths.gps_data)[
                    "Range [km]"
                ].values.tolist()[START:END],
                tilt=pd.read_csv(SWELLEX96Paths.gps_data)[
                    "Apparent Tilt [deg]"
                ].values.tolist()[START:END],
            ),
        ),
        fixed=builds(dict, src_z=60.0),
    ),
    mode="simulation",
    objective=SimulationObjectiveConf,
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
                time_step=range(0, 125)[START:END],
                rec_r=pd.read_csv(SWELLEX96Paths.gps_data)[
                    "Apparent Range [km]"
                ].values.tolist()[START:END],
                tilt=pd.read_csv(SWELLEX96Paths.gps_data)[
                    "Apparent Tilt [deg]"
                ].values.tolist()[START:END],
            ),
        ),
        fixed=builds(dict, src_z=60.0),
    ),
    mode="experimental",
    objective=ExperimentalObjectiveConf,
    mc_config=ExperimentalMonteCarloConf,
    builds_bases=(ParameterizationConf,),
)


# PROBLEM TYPE CONFIG
@dataclass
class ProblemDefinition:
    name: str
    search_space: SearchConf


ProblemDefinitionConf = builds(
    ProblemDefinition,
    name="loc_tilt",
    search_space=SearchConf,
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
cs.store(group="problem", name="loc_tilt", node=ProblemDefinitionConf)
cs.store(group="metric", name="bartlett", node=BartlettMetricConf)
cs.store(group="metric", name="bartlett_ml", node=BartlettMLMetricConf)
