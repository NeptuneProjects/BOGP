# BOGP - EI
python projects/swellex96_inv/data/bo/run.py --init=200 --simulate
python projects/swellex96_inv/data/bo/run.py --init=400
python projects/swellex96_inv/data/bo/run.py --init=300
python projects/swellex96_inv/data/bo/run.py --init=200
python projects/swellex96_inv/data/bo/run.py --init=100
python projects/swellex96_inv/data/bo/run.py --init=50

# BOGP - UCB
python projects/swellex96_inv/data/bo/run.py --optim=ucb --init=200 --serial=sim_ucb_b1 --simulate
python projects/swellex96_inv/data/bo/run.py --optim=ucb --init=200 --serial=exp_ucb_b1

# SOBOL
python projects/swellex96_inv/data/bo/run.py --optim=sobol --init=200 --serial=sim_sobol --budget=50000 --simulate
python projects/swellex96_inv/data/bo/run.py --optim=sobol --init=200 --serial=exp_sobol --budget=50000