#!/bin/bash
# 
# Script to aggregate the results of the swellex96 simulation
# 
# Usage:
# zsh ./scripts/swellex96/agg.sh

# python3 data/aggregate.py ../data/swellex96_S5_VLA/outputs/localization/simulation/serial_000
python3 projects/swellex96_localization/data/aggregate.py ../data/swellex96_S5_VLA/outputs/localization/experimental/serial_001
