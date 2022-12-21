#!/bin/bash
# 
# Runs matched field processing script. Sample usage:
# bash ./Source/scripts/mfp_run.sh

# MFP for figures:
# python3 ./Source/scripts/mfp.py '/Users/williamjenkins/Research/Projects/BOGP/Data/SWELLEX96/ambiguity_surfaces/' --dr 0.05 --dz 2

# MFP with restricted evaluation budget
NR=40
NZ=30
echo "Running MFP with $NR range points, $NZ depth points, $(( NR * NZ )) total evaluations."
python3 ./Source/scripts/mfp.py '/Users/williamjenkins/Research/Projects/BOGP/Data/Simulations/working/mfp/' --nr $NR --nz $NZ