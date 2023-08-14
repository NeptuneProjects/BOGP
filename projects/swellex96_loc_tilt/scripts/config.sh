#!/bin/bash
# 
# Example usage:
# zsh ./projects/swellex96_loc_tilt/scripts/config.sh serial_002 simulation
# zsh ./projects/swellex96_loc_tilt/scripts/config.sh serial_006 experimental

SERIAL="$1"
MODE="$2"

python3 ./projects/swellex96_loc_tilt/data/configure.py serial=$SERIAL parameterization=$MODE "optimizers=[gpei]"
# python3 ./projects/swellex96_loc_tilt/data/configure.py serial=$SERIAL parameterization=$MODE "optimizers=[qgpei]"
