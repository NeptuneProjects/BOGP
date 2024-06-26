# RANDOM
python projects/swellex96_inv/data/bo/run.py --optim=random --init=32 --budget=10000 --serial=sim_random --simulate
python projects/swellex96_inv/data/bo/run.py --optim=random --init=32 --budget=10000 --serial=exp_random

# BOGP - UCB
python projects/swellex96_inv/data/bo/run.py --optim=ucb --init=64 --serial=sim_ucb --simulate
python projects/swellex96_inv/data/bo/run.py --optim=ucb --init=64 --serial=exp_ucb

# BOGP - EI
python projects/swellex96_inv/data/bo/run.py --optim=ei --init=64 --serial=sim_ei --simulate
python projects/swellex96_inv/data/bo/run.py --optim=ei --init=64 --serial=exp_ei

# BOGP - LogEI
python projects/swellex96_inv/data/bo/run.py --optim=logei --init=64 --serial=sim_logei --simulate
python projects/swellex96_inv/data/bo/run.py --optim=logei --init=96 --serial=exp_logei
python projects/swellex96_inv/data/bo/run.py --optim=logei --init=64 --serial=exp_logei
python projects/swellex96_inv/data/bo/run.py --optim=logei --init=32 --serial=exp_logei
python projects/swellex96_inv/data/bo/run.py --optim=logei --init=16 --serial=exp_logei
python projects/swellex96_inv/data/bo/run.py --optim=logei --init=8 --serial=exp_logei
