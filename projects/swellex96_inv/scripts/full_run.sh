# SOBOL
python projects/swellex96_inv/data/bo/run.py --optim=sobol --budget=50000 --serial=exp_sobol
python projects/swellex96_inv/data/bo/run.py --optim=sobol --budget=50000 --serial=sim_sobol --simulate

# BOGP - UCB
python projects/swellex96_inv/data/bo/run.py --optim=ucb --init=200 --serial=sim_ucb --simulate
python projects/swellex96_inv/data/bo/run.py --optim=ucb --init=200 --serial=exp_ucb

# BOGP - EI
python projects/swellex96_inv/data/bo/run.py --optim=ei --init=200 --serial=sim_ei --simulate
python projects/swellex96_inv/data/bo/run.py --optim=ei --init=200 --serial=exp_ei
python projects/swellex96_inv/data/bo/run.py --optim=ei --init=400 --serial=exp_ei
python projects/swellex96_inv/data/bo/run.py --optim=ei --init=300 --serial=exp_ei
python projects/swellex96_inv/data/bo/run.py --optim=ei --init=100 --serial=exp_ei
python projects/swellex96_inv/data/bo/run.py --optim=ei --init=50 --serial=exp_ei