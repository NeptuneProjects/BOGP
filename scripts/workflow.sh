#!/bin/bash

bash ./Source/scripts/config.sh ./Source/scripts/config.toml r s
bash ./Source/scripts/run.sh r s serial_230217 cpu 12
python3 ./Source/BOGP/aggregate.py ./Data/localization/experimental/serial_full_depth
