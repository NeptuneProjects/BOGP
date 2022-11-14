#!/bin/bash
# 
# Runs configuration script. Sample usage:
# bash ./Source/scripts/sim_agg_results.sh

python3 ./Source/scripts/aggregate.py r l
python3 ./Source/scripts/aggregate.py s l