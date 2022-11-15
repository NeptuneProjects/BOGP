#!/bin/bash
# 
# Runs configuration and production scripts. Sample usage:
# bash ./Source/scripts/sim_run.sh l 4 cuda 2
# bash ./Source/scripts/sim_run.sh l 4 cpu 2

SIM="$1"
JOBS=$2
DEVICE=$(echo "$3" | awk '{print toupper($0)}')
OFFSET=${4:-0}

if [[ $SIM == "r" ]]
    then
        SIMDESC="range estimation"
        SIMPATH="range_estimation"
elif [[ $SIM == "l" ]]
    then
        SIMDESC="localization"
        SIMPATH=$SIMDESC
fi

echo "Running $SIMDESC on $DEVICE with $JOBS jobs."
if [[ $DEVICE == "CPU" ]]
    then
        find ./Data/Simulations/$SIMPATH/queue/*.json \
        | parallel -j$JOBS --progress \
        'python3 ./Source/scripts/simulations.py '$SIM' {} --path ./Data/Simulations'
elif [[ $DEVICE == "CUDA" ]]
    then
        find ./Data/Simulations/$SIMPATH/queue/*.json \
        | parallel -j$JOBS --progress \
        'CUDA_VISIBLE_DEVICES="$(({%}-1+'$OFFSET'))" python3 ./Source/scripts/simulations.py '$SIM' {} --path ./Data/Simulations --device cuda'
fi
