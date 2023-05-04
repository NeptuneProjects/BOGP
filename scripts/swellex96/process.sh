#!/bin/bash

python3 ./data/swellex96/acoustics.py process --freqs 148,166,201,235,283,338,388
python3 ./data/swellex96/acoustics.py ambsurf --freqs 148,166,201,235,283,338,388
python3 ./data/swellex96/acoustics.py multifreq --freqs 148,166,201,235,283,338,388