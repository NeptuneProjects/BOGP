#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import asdict, dataclass, field
from typing import Type, Optional, Union

from ax.modelbridge.base import ModelBridge
from ax.modelbridge.generation_strategy import GenerationStep
from botorch.acquisition import ExpectedImprovement
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood


@dataclass
class OptimizerKwargs:
    num_restarts: Optional[int] = None
    raw_samples: Optional[int] = None


@dataclass
class ModelGenOptions:
    optimizer_kwargs: Optional[OptimizerKwargs] = None


@dataclass
class ModelGenKwargs:
    model_gen_options: Optional[ModelGenOptions] = None


@dataclass
class SurrogateKwargs:
    botorch_model_class: Optional[Type[SingleTaskGP]] = None
    mll_class: Optional[MarginalLogLikelihood] = None


@dataclass
class ModelKwargs:
    surrogate: SurrogateKwargs = field(default_factory=SurrogateKwargs)
    botorch_acqf_class: AcquisitionFunction = field(default=ExpectedImprovement)


@dataclass
class GenerationStepKwargs:
    model: ModelBridge
    num_trials: int
    max_parallelism: Optional[int] = None
    use_update: bool = False
    enforce_num_trials: bool = True
    should_deduplicate: bool = False
    model_kwargs: ModelKwargs = field(default_factory=ModelKwargs)
    model_gen_kwargs: ModelGenKwargs = field(default_factory=ModelGenKwargs)


def construct_generation_step(
    model: ModelBridge,
    num_trials: int,
    botorch_model_class: Optional[Type[SingleTaskGP]] = None,
    mll_class: Optional[MarginalLogLikelihood] = None,
    botorch_acqf_class: Optional[AcquisitionFunction] = None,
    max_parallelism: Optional[int] = None,
    use_update: bool = False,
    enforce_num_trials: bool = True,
    should_deduplicate: bool = False,
    num_restarts: Optional[int] = None,
    raw_samples: Optional[int] = None,
) -> GenerationStep:
    """Construct a GenerationStep object from a flat dictionary."""
    model_gen_kwargs = construct_model_gen_kwargs(num_restarts, raw_samples)
    model_kw = construct_model_kw(botorch_model_class, mll_class, botorch_acqf_class)
    gs_kwargs = GenerationStepKwargs(
        model=model,
        num_trials=num_trials,
        model_kwargs=model_kw,
        model_gen_kwargs=model_gen_kwargs,
        max_parallelism=max_parallelism,
        use_update=use_update,
        enforce_num_trials=enforce_num_trials,
        should_deduplicate=should_deduplicate,
    )
    return GenerationStep(**asdict(gs_kwargs))


def construct_model_gen_kwargs(
    num_restarts: Optional[int] = None, raw_samples: Optional[int] = None
) -> Union[ModelGenKwargs, None]:
    if (not num_restarts) and (not raw_samples):
        return None
    return ModelGenKwargs(
        model_gen_options=ModelGenOptions(
            optimizer_kwargs=OptimizerKwargs(num_restarts, raw_samples)
        )
    )


def construct_model_kw(
    botorch_model_class=None, mll_class=None, botorch_acqf_class=None
):
    if (not botorch_model_class) and (not mll_class) and (not botorch_acqf_class):
        return None
    return ModelKwargs(
        surrogate=SurrogateKwargs(
            botorch_model_class=botorch_model_class, mll_class=mll_class
        ),
        botorch_acqf_class=botorch_acqf_class,
    )
