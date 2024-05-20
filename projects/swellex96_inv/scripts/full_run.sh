# BOGP - UCB
# python projects/swellex96_inv/data/bo/run.py --optim=ucb --init=32 --serial=sim_ucb --simulate
python projects/swellex96_inv/data/bo/run.py --optim=ucb --init=32 --serial=exp_ucb

# BOGP - EI
python projects/swellex96_inv/data/bo/run.py --optim=ei --init=32 --serial=sim_ei --simulate
python projects/swellex96_inv/data/bo/run.py --optim=ei --init=32 --serial=exp_ei
# python projects/swellex96_inv/data/bo/run.py --optim=ei --init=64 --serial=exp_ei
# python projects/swellex96_inv/data/bo/run.py --optim=ei --init=16 --serial=exp_ei
# python projects/swellex96_inv/data/bo/run.py --optim=ei --init=8 --serial=exp_ei

# BOGP - LogEI
python projects/swellex96_inv/data/bo/run.py --optim=logei --init=32 --serial=sim_logei --simulate
python projects/swellex96_inv/data/bo/run.py --optim=logei --init=32 --serial=exp_logei

# SOBOL
python projects/swellex96_inv/data/bo/run.py --optim=sobol --budget=50000 --serial=sim_sobol --simulate
python projects/swellex96_inv/data/bo/run.py --optim=sobol --budget=50000 --serial=exp_sobol