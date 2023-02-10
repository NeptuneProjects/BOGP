#!/bin/bash

bash ./Source/scripts/config.sh ./Source/scripts/config.toml l e
bash ./Source/scripts/run.sh r e serial_002 cpu 12
python3 ./Source/BOGP/aggregate.py ./Data/range_estimation/experimental/serial_002
