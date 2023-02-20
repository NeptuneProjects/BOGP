#!/bin/bash

bash ./Source/scripts/config.sh ./Source/scripts/config.toml r s
bash ./Source/scripts/run.sh r s serial_230217 cpu 12
python3 ./Source/BOGP/aggregate.py ./Data/range_estimation/simulation/serial_230217
