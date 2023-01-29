#!/bin/bash
# 
# Usage:
# bash ./Source/scripts/run.sh r e serial_20230129T104000 cuda 6 2

OPTIM="$1" # [(r)ange estimation, (l)ocalization]
MODE="$2"  # [(s)imulation, (e)xperimental]
SERIAL="$3"
DEVICE=$(echo "$4" | awk '{print toupper($0)}')
JOBS=$5
OFFSET=${6:-0}

if [[ $OPTIM == "r" ]]
    then
        OPTIMDESC="range estimation"
        OPTIMPATH="range_estimation"
elif [[ $OPTIM == "l" ]]
    then
        OPTIMDESC="localization"
        OPTIMPATH=$OPTIMDESC
fi
if [[ $MODE == "s" ]]
    then
        MODEDESC="simulation"
elif [[ $MODE == "e" ]]
    then
        MODEDESC="experimental"
fi
MODEPATH=$MODEDESC

echo "Running $OPTIMDESC ($MODEDESC) on $DEVICE with $JOBS jobs."
if [[ $DEVICE == "CPU" ]]
    then
        find ./Data/$OPTIMPATH/$MODEPATH/$SERIAL/queue/*.json \
        | parallel -j$JOBS --progress \
        'python3 ./Source/BOGP/runner.py {}'
elif [[ $DEVICE == "CUDA" ]]
    then
        find ./Data/$OPTIMPATH/$MODEPATH/$SERIAL/queue/*.json \
        | parallel -j$JOBS --progress \
        'CUDA_VISIBLE_DEVICES="$(({%}-1+'$OFFSET'))" python3 ./Source/BOGP/runner.py {} --device cuda'
fi
