#!/bin/bash
# 
# Runs configuration script. Sample usage:
# For range estimation:
# bash ./Source/scripts/sim_agg_results.sh r
# For localization:
# bash ./Source/scripts/sim_agg_results.sh l

SIM="$1"

python3 ./Source/scripts/aggregate.py r $SIM
python3 ./Source/scripts/aggregate.py s $SIM