#!/bin/bash
# 
# Runs configuration script. Sample usage:
# bash ./Source/scripts/sim_config.sh l

SIM="$1"
if [[ $SIM == "r" ]]
    then
        SIMDESC="range estimation"
elif [[ $SIM == "l" ]]
    then
        SIMDESC="localization"
fi
python3 ./Source/scripts/configurations.py $SIM
echo "$SIMDESC simulation configured."
