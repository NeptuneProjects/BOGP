#!/bin/bash

# Collate the data from the swellex96 experiment.
# Usage:
#   $ cd swellex96
#   $ zsh ./projects/swellex96_loc_tilt/scripts/collate.sh

python projects/swellex96_loc_tilt/data/collate.py experimental serial_003
