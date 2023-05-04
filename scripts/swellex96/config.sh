#!/bin/bash
# 
# Example usage:
# zsh ./scripts/swellex96/config.sh serial_000 simulation
# zsh ./scripts/swellex96/config.sh serial_000 experimental

SERIAL="$1"
MODE="$2"

python3 ./data/swellex96/configure.py serial=$SERIAL parameterization=$MODE