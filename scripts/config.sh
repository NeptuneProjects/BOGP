#!/bin/bash
# 
# Usage:
# bash ./Source/scripts/config.sh ./Source/scripts/config.toml rl
# bash ./Source/scripts/config.sh ./Source/scripts/config.toml rl YYMMDD_DESC

CONFIGPATH=$1
OPTIMS=$2
MODES=$3
SERIAL=${4:-}

if [ -z "$4" ]
    then
        python3 ./Source/BOGP/configure.py $CONFIGPATH $OPTIMS $MODES
    else
        python3 ./Source/BOGP/configure.py $CONFIGPATH $OPTIMS $MODES --serial $SERIAL
fi
