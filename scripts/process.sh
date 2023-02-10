#!/bin/bash

python3 ./Source/BOGP/acoustics.py process --freqs 148,166,201,235,283,338,388
python3 ./Source/BOGP/acoustics.py ambsurf --freqs 148,166,201,235,283,338,388
python3 ./Source/BOGP/acoustics.py multifreq --freqs 148,166,201,235,283,338,388