#!/bin/bash
# 
# Script to aggregate the results of the swellex96 simulation
# 
# Usage:
# zsh ./projects/swellex96_loc_tilt/scripts/agg.sh ../data/swellex96_S5_VLA_loc_tilt/outputs/loc_tilt/experimental/serial_005

DIR="$1"
MODE=$(basename $(dirname "$DIR"))
SERIAL=$(basename $DIR)

echo "Loading $MODE results for $SERIAL."

python3 optimization/aggregate.py $DIR && \
python3 projects/swellex96_loc_tilt/data/collate.py $MODE $SERIAL
