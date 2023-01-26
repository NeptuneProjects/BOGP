#!/bin/bash
# 
# Usage:
# bash ./Source/scripts/config.sh ./Source/scripts/config.toml rl
# bash ./Source/scripts/config.sh ./Source/scripts/config.toml rl YYMMDD_DESC

CONFIGPATH=$1
OPTIMS=$2
SERIAL=${3:-}

if [ -z "$3" ]
    then
        python3 ./Source/BOGP/configure.py $CONFIGPATH $OPTIMS
    else
        python3 ./Source/BOGP/configure.py $CONFIGPATH $OPTIMS --serial $SERIAL
fi
