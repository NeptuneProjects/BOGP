#!/bin/bash

python3 ./data/swellex96/env.py \
    ../data/swellex96_S5_VLA/ctd/i9606.prn \
    ../data/swellex96_S5_VLA/env_models/swellex96.json \
    --title swellex96 \
    --model KRAKEN
