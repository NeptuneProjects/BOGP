#!/bin/bash
#
# Builds acoustic environment model for swellex96

python3 ./projects/swellex96_localization/data/env.py \
    ../data/swellex96_S5_VLA/ctd/i9606.prn \
    ../data/swellex96_S5_VLA/env_models/swellex96.json \
    --title swellex96 \
    --model KRAKEN \
    # --tilt -1.0
