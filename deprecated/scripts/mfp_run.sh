#!/bin/bash
# 
# Runs matched field processing script. Sample usage:
# bash ./Source/scripts/mfp_run.sh

# MFP for figures:
# python3 ./Source/scripts/mfp.py '/Users/williamjenkins/Research/Projects/BOGP/Data/SWELLEX96/ambiguity_surfaces/' --dr 0.05 --dz 2

# MFP range estimation with restricted evaluation budget
# NR=200
# echo "Running MFP with $NR range points."
# python3 ./Source/scripts/mfp.py '/Users/williamjenkins/Research/Projects/BOGP/Data/Simulations/working/mfp/' r --nr $NR

# MFP localization with restricted evaluation budget
NR=50
NZ=20
echo "Running MFP with $NR range points, $NZ depth points, $(( NR * NZ )) total evaluations."
python3 ./Source/scripts/mfp.py '/Users/williamjenkins/Research/Projects/BOGP/Data/Simulations/working/mfp/' l --nr $NR --nz $NZ