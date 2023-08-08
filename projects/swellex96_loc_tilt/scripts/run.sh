#!/bin/bash
# 
# Example usage:
# zsh ./projects/swellex96_loc_tilt/scripts/run.sh ../../data/swellex96_S5_VLA_loc_tilt/outputs/loc_tilt/experimental/serial_001/queue 4
# zsh ./projects/swellex96_loc_tilt/scripts/run.sh ../data/swellex96_S5_VLA_loc_tilt/outputs/loc_tilt/simulation/serial_001/queue 4

QUEUE="$1"
JOBS=$2

find $QUEUE/*.yaml \
    | parallel -j$JOBS --progress \
    'HYDRA_FULL_ERROR=1 python3 projects/swellex96_loc_tilt/data/run.py --config-path ../../../'$QUEUE' --config-name $(basename ${})'
