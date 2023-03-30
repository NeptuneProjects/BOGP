#!/bin/bash

bash ./Source/scripts/config.sh ./Source/scripts/config.toml l s
bash ./Source/scripts/run.sh l s serial_multifreq_full cpu 12
python3 ./Source/BOGP/aggregate.py ./Data/localization/simulation/serial_multifreq_full
python3 ./Source/BOGP/collate.py l s serial_multifreq_full3