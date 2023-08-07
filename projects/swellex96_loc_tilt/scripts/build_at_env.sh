#!/bin/bash
#
# Builds acoustic environment model for swellex96
# Usage:
# bash ./projects/swellex96_loc_tilt/scripts/build_at_env.sh

python3 ./projects/swellex96_loc_tilt/data/env.py \
    ../data/swellex96_S5_VLA_loc_tilt/ctd/i9606.prn \
    ../data/swellex96_S5_VLA_loc_tilt/env_models/swellex96.json \
    --title swellex96 \
    --model KRAKEN \
    # --tilt -1.0
