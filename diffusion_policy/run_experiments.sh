#!/bin/bash

# Run experiments

python3 diffusion_policy/rollout.py --task lift --n_runs 20
python3 diffusion_policy/rollout.py --task can --n_runs 20
python3 diffusion_policy/rollout.py --task square --n_runs 20

python3 diffusion_policy/rollout.py --task lift --kp 1000 --kp_limits 0 2000 --damping_limits 0 67 --n_runs 20
python3 diffusion_policy/rollout.py --task can --kp 1000 --kp_limits 0 2000 --damping_limits 0 67 --n_runs 20
python3 diffusion_policy/rollout.py --task square --kp 1000 --kp_limits 0 2000 --damping_limits 0 67 --n_runs 20

# python3 diffusion_policy/rollout.py --task lift --kp 1500 --kp_limits 0 3000 --damping_limits 0 100 --n_runs 20
# python3 diffusion_policy/rollout.py --task can --kp 1500 --kp_limits 0 3000 --damping_limits 0 100 --n_runs 20
# python3 diffusion_policy/rollout.py --task square --kp 1500 --kp_limits 0 3000 --damping_limits 0 100 --n_runs 20

python3 diffusion_policy/rollout.py --task lift --n_runs 20 --use_awe
python3 diffusion_policy/rollout.py --task can --n_runs 20 --use_awe
python3 diffusion_policy/rollout.py --task square --n_runs 20 --use_awe

python3 diffusion_policy/rollout.py --task lift --kp 1000 --kp_limits 0 2000 --damping_limits 0 67 --n_runs 20 --use_awe
python3 diffusion_policy/rollout.py --task can --kp 1000 --kp_limits 0 2000 --damping_limits 0 67 --n_runs 20 --use_awe
python3 diffusion_policy/rollout.py --task square --kp 1000 --kp_limits 0 2000 --damping_limits 0 67 --n_runs 20 --use_awe

# python3 diffusion_policy/rollout.py --task lift --kp 1500 --kp_limits 0 3000 --damping_limits 0 100 --n_runs 20 --use_awe
# python3 diffusion_policy/rollout.py --task can --kp 1500 --kp_limits 0 3000 --damping_limits 0 100 --n_runs 20 --use_awe
# python3 diffusion_policy/rollout.py --task square --kp 1500 --kp_limits 0 3000 --damping_limits 0 100 --n_runs 20 --use_awe