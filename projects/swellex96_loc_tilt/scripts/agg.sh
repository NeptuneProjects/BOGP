#!/bin/bash
# 
# Script to aggregate the results of the swellex96 simulation
# 
# Usage:
# zsh ./projects/swellex96_loc_tilt/scripts/agg.sh

# python3 data/aggregate.py ../data/swellex96_S5_VLA/outputs/localization/simulation/serial_000
python3 optimization/aggregate.py ../data/swellex96_S5_VLA_loc_tilt/outputs/loc_tilt/experimental/serial_003
