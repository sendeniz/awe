#!/bin/bash

# Run experiments

# originale experiments from repo default multiplier
# default kp: 
python3 diffusion_policy/rollout.py --task lift --n_runs 20 --multiplier 1
python3 diffusion_policy/rollout.py --task can --n_runs 20 --multiplier 1
python3 diffusion_policy/rollout.py --task square --n_runs 20 --multiplier 1

python3 diffusion_policy/rollout.py --task lift --n_runs 20 --use_awe --multiplier 10
python3 diffusion_policy/rollout.py --task can --n_runs 20 --use_awe --multiplier 10
python3 diffusion_policy/rollout.py --task square --n_runs 20 --use_awe --multiplier 10


# high kp gains: default multiplier
python3 diffusion_policy/rollout.py --task lift --kp 1000 --kp_limits 0 2000 --damping_limits 0 67 --n_runs 20 --multiplier 1 
python3 diffusion_policy/rollout.py --task can --kp 1000 --kp_limits 0 2000 --damping_limits 0 67 --n_runs 20 --multiplier 1 
python3 diffusion_policy/rollout.py --task square --kp 1000 --kp_limits 0 2000 --damping_limits 0 67 --n_runs 20 --multiplier 1 

python3 diffusion_policy/rollout.py --task lift --kp 1000 --kp_limits 0 2000 --damping_limits 0 67 --n_runs 20 --use_awe --multiplier 10
python3 diffusion_policy/rollout.py --task can --kp 1000 --kp_limits 0 2000 --damping_limits 0 67 --n_runs 20 --use_awe --multiplier 10
python3 diffusion_policy/rollout.py --task square --kp 1000 --kp_limits 0 2000 --damping_limits 0 67 --n_runs 20 --use_awe --multiplier 10 
