#!/bin/bash
# 
# Example usage:
# zsh ./scripts/swellex96/run.sh ../data/swellex96_S5_VLA/outputs/localization/experimental/serial_000/queue 4
# zsh ./scripts/swellex96/run.sh ../data/swellex96_S5_VLA/outputs/localization/simulation/serial_001/queue 1

QUEUE="$1"
JOBS=$2

find $QUEUE/*.yaml \
    | parallel -j$JOBS --progress \
    'HYDRA_FULL_ERROR=1 python3 data/swellex96/run.py --config-path ../../'$QUEUE' --config-name $(basename ${})'
