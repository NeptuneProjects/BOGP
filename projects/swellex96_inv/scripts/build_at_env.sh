#!/bin/bash
#
# Builds acoustic environment model for swellex96
# Usage:
# bash ./projects/swellex96_inv/scripts/build_at_env.sh

python3 ./projects/swellex96_inv/data/env.py \
    ../data/swellex96_S5_VLA_inv/ctd/i9605.prn \
    ../data/swellex96_S5_VLA_inv/env_models/swellex96.json \
    --title swellex96 \
    --model KRAKENC \
    # --tilt -1.0
