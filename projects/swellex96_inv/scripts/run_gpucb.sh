# To run:

# CUDA_VISIBLE_DEVICES=0 python projects/swellex96_inv/data/bo/run.py --optim=ucb --init=400 --serial=exp_ucb_b1
# CUDA_VISIBLE_DEVICES=0 python projects/swellex96_inv/data/bo/run.py --optim=ucb --init=300 --serial=exp_ucb_b1
CUDA_VISIBLE_DEVICES=0 python projects/swellex96_inv/data/bo/run.py --optim=ucb --init=200 --serial=sim_ucb_b1 --simulate
# CUDA_VISIBLE_DEVICES=0 python projects/swellex96_inv/data/bo/run.py --optim=ucb --init=100 --serial=exp_ucb_b1
# CUDA_VISIBLE_DEVICES=0 python projects/swellex96_inv/data/bo/run.py --optim=ucb --init=50 --serial=exp_ucb_b1

# CUDA_VISIBLE_DEVICES=0 python projects/swellex96_inv/data/bo/run.py --optim=ucb --beta=0.1 --init=400 --serial=exp_ucb_b01
# CUDA_VISIBLE_DEVICES=0 python projects/swellex96_inv/data/bo/run.py --optim=ucb --beta=0.1 --init=300 --serial=exp_ucb_b01
CUDA_VISIBLE_DEVICES=0 python projects/swellex96_inv/data/bo/run.py --optim=ucb --beta=0.1 --init=200 --serial=sim_ucb_b01 --simulate
# CUDA_VISIBLE_DEVICES=0 python projects/swellex96_inv/data/bo/run.py --optim=ucb --beta=0.1 --init=100 --serial=exp_ucb_b01
# CUDA_VISIBLE_DEVICES=0 python projects/swellex96_inv/data/bo/run.py --optim=ucb --beta=0.1 --init=50 --serial=exp_ucb_b01

# CUDA_VISIBLE_DEVICES=0 python projects/swellex96_inv/data/bo/run.py --optim=ucb --beta=5 --init=400 --serial=exp_ucb_b5
# CUDA_VISIBLE_DEVICES=0 python projects/swellex96_inv/data/bo/run.py --optim=ucb --beta=5 --init=300 --serial=exp_ucb_b5
CUDA_VISIBLE_DEVICES=0 python projects/swellex96_inv/data/bo/run.py --optim=ucb --beta=5 --init=200 --serial=sim_ucb_b5 --simulate
# CUDA_VISIBLE_DEVICES=0 python projects/swellex96_inv/data/bo/run.py --optim=ucb --beta=5 --init=100 --serial=exp_ucb_b5
# CUDA_VISIBLE_DEVICES=0 python projects/swellex96_inv/data/bo/run.py --optim=ucb --beta=5 --init=50 --serial=exp_ucb_b5


# CUDA_VISIBLE_DEVICES=0 python projects/swellex96_inv/data/bo/run.py --optim=ucb --beta=0.1 --init=200 --serial=exp_ucb_b01
# CUDA_VISIBLE_DEVICES=0 python projects/swellex96_inv/data/bo/run.py --optim=ucb --beta=5 --init=200 --serial=exp_ucb_b5

# In-progress
CUDA_VISIBLE_DEVICES=1 python projects/swellex96_inv/data/bo/run.py --optim=ucb --init=200 --serial=exp_ucb_b1



# Completed
