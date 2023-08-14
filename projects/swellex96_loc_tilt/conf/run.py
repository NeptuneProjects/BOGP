#!/usr/bin/env python
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from pathlib import Path
import sys

from hydra.core.config_store import ConfigStore
from hydra_zen import (
    builds,
    instantiate,
    make_custom_builds_fn,
    MISSING,
)
from oao.objective import (
    NoiselessFormattedObjective,
    NoisyFormattedObjective,
    Objective,
)
from oao.optimizer import BayesianOptimization, GridSearch
from oao.space import SearchParameterBounds, SearchSpaceBounds
from omegaconf import DictConfig, OmegaConf
from tritonoa.at.models.kraken.runner import run_kraken
from tritonoa.sp.beamforming import beamformer
from tritonoa.sp.mfp import MatchedFieldProcessor

from conf.common import SWELLEX96Paths

sys.path.insert(0, str(Path(__file__).parents[3]))
from optimization.strategy import GPEIStrategy, SobolStrategy, GridSearchStrategy
from optimization import utils


def format_objective(cfg: DictConfig, objective: Objective):
    return instantiate(cfg.formatter)(
        objective=objective,
        name=cfg.metric_name,
        properties_kw={"minimize": cfg.minimize_objective},
    )


def instantiate_experimental_covariance(cfg: DictConfig):
    return instantiate(
        cfg.objective.covariance_matrix.conf,
    )(
        paths=instantiate(cfg.objective.covariance_matrix.conf.paths)(
            freq=cfg.frequencies,
            path=SWELLEX96Paths.acoustic_path,
        ),
        index=instantiate(cfg.objective.covariance_matrix.conf.index)(
            key="time_step",
            parameterization=OmegaConf.to_object(cfg.parameterization),
        ),
    )


def instantiate_simulation_covariance(cfg: DictConfig):
    return instantiate(cfg.objective.covariance_matrix.conf)(
        runner=instantiate(cfg.objective.runner),
        parameters=instantiate(cfg.objective.covariance_matrix.conf.parameters)
        | OmegaConf.to_object(cfg.parameterization),
        freq=cfg.frequencies,
    )


def instantiate_objective(cfg: DictConfig):
    if cfg.objective.covariance_matrix.name == "simulation":
        covariance_matrix = instantiate_simulation_covariance(cfg)
    if cfg.objective.covariance_matrix.name == "experimental":
        covariance_matrix = instantiate_experimental_covariance(cfg)
    return instantiate(
        cfg.objective,
    )(
        runner=instantiate(cfg.objective.runner),
        covariance_matrix=covariance_matrix,
        freq=cfg.frequencies,
        parameters=instantiate(cfg.objective.parameters) | {"tmpdir": cfg.savepath},
        beamformer=instantiate(cfg.objective.beamformer),
    )


# Custom build functions
sbuilds = make_custom_builds_fn(populate_full_signature=True)
pbuilds = make_custom_builds_fn(populate_full_signature=False, zen_partial=True)

# objective
MFPConf = pbuilds(
    MatchedFieldProcessor,
    runner=MISSING,
    covariance_matrix=MISSING,
    freq=MISSING,
    parameters=MISSING,
    beamformer=MISSING,
    multifreq_method=MISSING,
)

# objective/runner
RunnerConf = pbuilds(run_kraken)


# objective/covariance_matrix
@dataclass
class CovarianceMatrixConfig:
    name: str
    conf: DictConfig


SimulatedCovarianceConf = pbuilds(
    CovarianceMatrixConfig,
    name="simulation",
    conf=pbuilds(
        utils.simulate_covariance,
        runner=MISSING,
        parameters=builds(
            utils.load_env_from_json, path=SWELLEX96Paths.environment_data
        ),
        freq=MISSING,
    ),
)

ExperimentalCovarianceConf = pbuilds(
    CovarianceMatrixConfig,
    name="experimental",
    conf=pbuilds(
        utils.load_covariance_matrices,
        paths=pbuilds(
            utils.get_covariance_matrix_paths,
            freq=MISSING,
            path=SWELLEX96Paths.acoustic_path,
        ),  # <- specify this
        index=pbuilds(
            utils.get_covariance_index_from_parameterization,
            key="time_step",
            parameterization=MISSING,
        ),
    ),
)

# objective/parameters
Swellex96EnvConf = builds(
    utils.load_env_from_json, path=SWELLEX96Paths.environment_data
)

# objective/beamformer
BartlettConf = pbuilds(beamformer)
BartlettMLConf = pbuilds(beamformer, atype="cbf_ml")


# formatter
NoiselessFormatConf = pbuilds(
    NoiselessFormattedObjective, name=MISSING, properties_kw=MISSING, return_type=float
)
NoisyFormatConf = pbuilds(
    NoisyFormattedObjective, name=MISSING, properties_kw=MISSING, return_type=float
)

# search_space
SearchConf = sbuilds(
    SearchSpaceBounds,
    bounds=[
        builds(
            SearchParameterBounds,
            name="rec_r",
            lower_bound=-0.5,
            upper_bound=0.5,
            relative=True,
            min_lower_bound=0.5,
            max_upper_bound=8.0,
            builds_bases=(SearchParameterBounds,),
        ),
        builds(
            SearchParameterBounds,
            name="src_z",
            lower_bound=-20.0,
            upper_bound=20.0,
            relative=True,
            min_lower_bound=1.0,
            max_upper_bound=200.0,
            builds_bases=(SearchParameterBounds,),
        ),
        builds(
            SearchParameterBounds,
            name="tilt",
            lower_bound=-5.0,
            upper_bound=5.0,
            relative=False,
            builds_bases=(SearchParameterBounds,),
        ),
    ],
    hydra_convert="object",
)

# strategy
GridSearchStrategyConf = builds(GridSearchStrategy, num_trials=[6, 6, 5])
SobolStrategyConf = pbuilds(
    SobolStrategy,
    num_trials=180,
    max_parallelism=16,
    seed=MISSING,
)
GPEIStrategyConf = pbuilds(
    GPEIStrategy,
    warmup_trials=128,
    warmup_parallelism=52,
    num_trials=16,
    max_parallelism=1,
    seed=MISSING,
)
qGPEIStrategyConf = pbuilds(
    GPEIStrategy,
    warmup_trials=128,
    warmup_parallelism=16,
    num_trials=52,
    max_parallelism=4,
    seed=MISSING,
)

# optimizer
GridSearchOptimizationConf = pbuilds(
    GridSearch, objective=MISSING, search_space=MISSING, strategy=GridSearchStrategyConf
)
SobolOptimizationConf = pbuilds(
    BayesianOptimization,
    objective=MISSING,
    search_space=MISSING,
    strategy=SobolStrategyConf,
)
GPEIOptimizationConf = pbuilds(
    BayesianOptimization,
    objective=MISSING,
    search_space=MISSING,
    strategy=GPEIStrategyConf,
)
qGPEIOptimizationConf = pbuilds(
    BayesianOptimization,
    objective=MISSING,
    search_space=MISSING,
    strategy=qGPEIStrategyConf,
)

# Configuration Store
cs = ConfigStore.instance()
cs.store(group="objective", name="mfp", node=MFPConf)
cs.store(group="objective/runner", name="kraken", node=RunnerConf)
cs.store(
    group="objective/covariance_matrix", name="simulation", node=SimulatedCovarianceConf
)
cs.store(
    group="objective/covariance_matrix",
    name="experimental",
    node=ExperimentalCovarianceConf,
)
cs.store(group="objective/parameters", name="swellex96", node=Swellex96EnvConf)
cs.store(group="objective/beamformer", name="bartlett", node=BartlettConf)
cs.store(group="objective/beamformer", name="bartlett_ml", node=BartlettMLConf)
cs.store(group="objective/multifreq_method", name="mean", node="mean")
cs.store(group="objective/multifreq_method", name="product", node="product")
cs.store(group="formatter", name="noiseless", node=NoiselessFormatConf)
cs.store(group="formatter", name="noisy", node=NoisyFormatConf)
cs.store(group="search_space", name="localization", node=SearchConf)
cs.store(group="optimizer", name="grid", node=GridSearchOptimizationConf)
cs.store(group="optimizer", name="sobol", node=SobolOptimizationConf)
cs.store(group="optimizer", name="gpei", node=GPEIOptimizationConf)
cs.store(group="optimizer", name="qgpei", node=qGPEIOptimizationConf)
