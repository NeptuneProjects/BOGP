#!/bin/bash
# 
# Runs configuration and production scripts. Sample usage:
# bash ./Source/scripts/workflow.sh l 4 cuda

SIM="$1"
JOBS=$2
DEVICE=$(echo "$3" | awk '{print toupper($0)}')
OFFSET=${4:-0}

if [[ $SIM == "r" ]]
    then
        SIMDESC="range estimation"
elif [[ $SIM == "l" ]]
    then
        SIMDESC="localization"
fi
python3 ./Source/scripts/configurations.py $SIM
echo "Running $SIMDESC on $DEVICE with $JOBS jobs."
if [[ $DEVICE == "CPU" ]]
    then
        find ./Data/Simulations/localization/queue/*.json \
        | parallel -j$JOBS --progress \
        'python3 ./Source/scripts/simulations.py '$SIM' {} --path ./Data/Simulations'
elif [[ $DEVICE == "CUDA" ]]
    then
        find ./Data/Simulations/localization/queue/*.json \
        | parallel -j$JOBS --progress \
        'CUDA_VISIBLE_DEVICES="$(({%}-1+'$OFFSET'))" python3 ./Source/scripts/simulations.py '$SIM' {} --path ./Data/Simulations --device cuda'
fi
python3 ./Source/scripts/aggregate.py r l
python3 ./Source/scripts/aggregate.py s l
# For debugging:
# find ./Data/Simulations/localization/queue/*.json | parallel -j1 --progress 'CUDA_VISIBLE_DEVICES=1 python3 ./Source/scripts/simulations.py l {} --path ./Data/Simulations --device cuda'