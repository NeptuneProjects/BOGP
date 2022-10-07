#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ast import Param
from inspect import Parameter
from ax import (
    ComparisonOp,
    ParameterType,
    RangeParameter,
    ChoiceParameter,
    FixedParameter,
    SearchSpace,
    Experiment,
    OutcomeConstraint,
    OrderConstraint,
    SumConstraint,
    OptimizationConfig,
    Objective,
    Metric,
    Runner
)
from ax.metrics.l2norm import L2NormMetric
from ax.metrics.hartmann6 import Hartmann6Metric
from ax.modelbridge.registry import Models
# import torch

# Define Parameter Space
NUM_DIM = 6

hartmann_search_space = SearchSpace(
    parameters=[
        RangeParameter(
            name=f"x{i}", parameter_type=ParameterType.FLOAT, lower=0.0, upper=1.0
        )
        for i in range(NUM_DIM)
    ]
)

choice_param = ChoiceParameter(
    name="choice",
    values=["foo", "bar"],
    parameter_type=ParameterType.STRING,
    is_ordered=False,
    sort_values=False
)
fixed_param = FixedParameter(
    name="fixed",
    value=[True],
    parameter_type=ParameterType.BOOL
)

# Define Constraints
sum_constraint = SumConstraint(
    parameters=[hartmann_search_space.parameters['x0'], hartmann_search_space.parameters['x1']],
    is_upper_bound=True,
    bound=5.0
)
order_constraint = OrderConstraint(
    lower_parameter=hartmann_search_space.parameters['x0'],
    upper_parameter=hartmann_search_space.parameters['x1']
)

# Create optimization config
param_names = [f"x{i}" for i in range(NUM_DIM)]
optimization_config = OptimizationConfig(
    objective=Objective(
        metric=Hartmann6Metric(name="hartmann6", param_names=param_names),
        minimize=True
    ),
    outcome_constraints=[
        OutcomeConstraint(
            metric=L2NormMetric(
                name="l2norm", param_names=param_names, noise_sd=0.2
            ),
            op=ComparisonOp.LEQ,
            bound=1.25,
            relative=False
        )
    ]
)

# Define a runner
class MyRunner(Runner):
    def run(self, trial):
        trial_metadata = {"name": str(trial.index)}
        return trial_metadata

# Create experiment
exp = Experiment(
    name="test_hartmann",
    search_space=hartmann_search_space,
    optimization_config=optimization_config,
    runner=MyRunner()
)

# Perform optimization
NUM_SOBOL_TRIALS = 5
NUM_BOTORCH_TRIALS = 15

print("Running Sobol initialization trials...")
sobol = Models.SOBOL(search_space=exp.search_space)

for i in range(NUM_SOBOL_TRIALS):
    # Produce a GeneratorRun from the model, which contains proposed 
    # arm(s) and other metadata
    generator_run = sobol.gen(n=1)
    # Add generator run to a trial to make it part of the epxeriment and
    # evaluate arm(s) in it
    trial = exp.new_trial(generator_run=generator_run)
    # Start trial run to evaluate arm(s) in the trial
    trial.run()
    # Mark trial as completed to record when a trial run is completed
    # and enable fetching of data for metrics on the experiment (by 
    # default, trials must be completed before metrics can fetch their 
    # data, unless a metric is explicitly configured otherwise)
    trial.mark_completed()

for i in range(NUM_BOTORCH_TRIALS):
    print(
        f"Running GP+EI optimization trial {i + NUM_SOBOL_TRIALS + 1}/{NUM_SOBOL_TRIALS + NUM_BOTORCH_TRIALS}..."
    )
    # Reinitialize GP+EI model at each step with updated data.
    gpei = Models.BOTORCH(experiment=exp, data=exp.fetch_data())
    generator_run = gpei.gen(n=1)
    trial = exp.new_trial(generator_run=generator_run)
    trial.run()
    trial.mark_completed()

print("Done!")

