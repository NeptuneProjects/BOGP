#!/bin/bash

# Collate the data from the swellex96 experiment.
# Usage:
#   $ cd swellex96
#   $ zsh ./scripts/swellex96/collate.sh

python projects/swellex96_localization/data/collate.py experimental serial_001
