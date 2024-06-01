#!/bin/bash

# HYDRA_FULL_ERROR=1 python3 ./projects/swellex96_inv/data/acoustics.py
HYDRA_FULL_ERROR=1 python3 ./projects/swellex96_inv/data/mfp.py
python3 ./projects/swellex96_inv/visualization/amb_surfaces.py --minimize

# Compute SNRs for frequencies used:
HYDRA_FULL_ERROR=1 python3 ./projects/swellex96_inv/data/acoustics.py --config-name="acoustics_snr"