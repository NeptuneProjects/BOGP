#!/bin/bash
# 
# Description of workflow
# Before running, ensure scripts 'configurations.py' and 'aggregate.py' 
# are configured for the desired workflow (i.e., specify acquisition 
# functions, parameters, etc.)
# 
# Usage:
# bash ./Source/scripts/workflow.sh

SIM="l"
JOBS=8
DEVICE="cpu"
OFFSET=0

zsh ./Source/scripts/sim_config.sh $SIM
zsh ./Source/scripts/sim_run.sh $SIM $JOBS $DEVICE $OFFSET
zsh ./Source/scripts/sim_agg_results.sh $SIM
