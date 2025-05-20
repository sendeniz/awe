import argparse
import json
import h5py
import imageio
import numpy as np
import os
from copy import deepcopy
import pprint

import torch

import robomimic
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.envs.env_base import EnvBase
from robomimic.algo import RolloutPolicy

from robomimic.envs.env_gym import EnvGym
import robosuite as suite

from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config
import matplotlib.pyplot as plt

import urllib.request

from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy
import hydra
from omegaconf import OmegaConf
from diffusion_policy.policy.diffusion_transformer_hybrid_image_policy import DiffusionTransformerHybridImagePolicy
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusion_policy.model.diffusion.ema_model import EMAModel

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
import os
import json
import cv2
import shutil

def move_task_results_to_run_folder(output_dir, task_name):
    """
    Move contents of the relevant task folder into the run-specific output directory.
    """
    task_to_folder = {
        "lift": "Lift",
        "square": "NutAssemblySquare",
        "can": "PickPlaceCan",
    }
    
    if (task_folder := task_to_folder.get(task_name)) and \
       os.path.exists(src_dir := os.path.join("results", task_folder)):
        
        # Move each item from task folder to run folder
        for item in os.listdir(src_dir):
            shutil.move(
                src=os.path.join(src_dir, item),
                dst=os.path.join(output_dir, item)
            )

def main(args):
    device = torch.device("cpu") #torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Construct checkpoint path based on arguments
    if args.use_awe:
        model_type = "awe"
        print("Initialising AWE based model.")
        ckpt_path = f"ckpts/{args.task}_awe_diffusion.ckpt"
        cfg = OmegaConf.load(f"config/waypoint_image_{args.task}_ph_diffusion_policy_transformer.yaml")
    else:
        model_type = "baseline"
        print("Initialising baseline model.")
        ckpt_path = f"ckpts/{args.task}_diffusion.ckpt"
        cfg = OmegaConf.load(f"config/baseline_image_{args.task}_ph_diffusion_policy_transformer.yaml")
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file {ckpt_path} does not exist")
    

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = OmegaConf.to_container(cfg, resolve=True)
    
    # Load policy once (outside the loop)
    policy_cfg = cfg["policy"]
    noise_scheduler = hydra.utils.instantiate(policy_cfg["noise_scheduler"]) 
    
    #DDPMScheduler(
    #    num_train_timesteps=policy_cfg["noise_scheduler"]["num_train_timesteps"],
    #    beta_start=policy_cfg["noise_scheduler"]["beta_start"],
    #    beta_end=policy_cfg["noise_scheduler"]["beta_end"],
    #    beta_schedule=policy_cfg["noise_scheduler"]["beta_schedule"],
    #    variance_type=policy_cfg["noise_scheduler"]["variance_type"],
    #    clip_sample=policy_cfg["noise_scheduler"]["clip_sample"],
    #    prediction_type=policy_cfg["noise_scheduler"]["prediction_type"],
    #)

    policy = hydra.utils.instantiate(policy_cfg, noise_scheduler=noise_scheduler)
    
    #DiffusionTransformerHybridImagePolicy(
    #    shape_meta=policy_cfg["shape_meta"],
    #    noise_scheduler=noise_scheduler,
    #    horizon=int(policy_cfg["horizon"]),
    #    n_action_steps=int(policy_cfg["n_action_steps"]),
    #    n_obs_steps=int(policy_cfg["n_obs_steps"]),
    #    num_inference_steps=int(policy_cfg["num_inference_steps"]),
    #    crop_shape=tuple(map(int, policy_cfg["crop_shape"])),
    #    obs_encoder_group_norm=policy_cfg["obs_encoder_group_norm"],
    #    eval_fixed_crop=policy_cfg["eval_fixed_crop"],
    #    n_layer=int(policy_cfg["n_layer"]),
    #    n_cond_layers=int(policy_cfg["n_cond_layers"]),
    #    n_head=int(policy_cfg["n_head"]),
    #    n_emb=int(policy_cfg["n_emb"]),
    #    p_drop_emb=float(policy_cfg["p_drop_emb"]),
    #    p_drop_attn=float(policy_cfg["p_drop_attn"]),
    #    causal_attn=policy_cfg["causal_attn"],
    #    time_as_cond=policy_cfg["time_as_cond"],
    #    obs_as_cond=policy_cfg["obs_as_cond"],
    #)
    
    ckpt = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(ckpt["state_dicts"]["ema_model"])
    policy.to(device)
    policy.eval()
    
    print(f"Running experiment {args.n_runs} times")
    
    for run in range(0, args.n_runs):
        print(f"\n=== Starting run {run+1}/{args.n_runs} ===")
        
        # Update configuration with command line arguments
        run_cfg = deepcopy(cfg)
        run_cfg["task"]["env_runner"]["controller_configs"]["kp"] = args.kp
        run_cfg["task"]["env_runner"]["controller_configs"]["kp_limits"] = args.kp_limits
        run_cfg["task"]["env_runner"]["controller_configs"]["damping_limits"] = args.damping_limits
        run_cfg["task"]["env_runner"]["n_envs"] = args.n_envs
        run_cfg["task"]["env_runner"]["n_test_vis"] = args.n_test_vis
        run_cfg["task"]["env_runner"]["n_test"] = args.n_test
        # different start seed for each run for higher variation between runs
        run_cfg["task"]["env_runner"]["test_start_seed"] = np.random.randint(0, 100000)

        # Create output directory for this run
        output_dir = os.path.join(f"results/{args.task}_{model_type}_kp_{args.kp}_run_{run}")
        os.makedirs(output_dir, exist_ok=True)
         
        # Create new runner for each run
        env_runner = hydra.utils.instantiate(
            run_cfg["task"]["env_runner"],
            output_dir=output_dir
        )
        
        # Run policy
        env_runner.run(policy)
        
        # Move task results to the run folder
        move_task_results_to_run_folder(output_dir, f"{args.task}")
        
        # free gpu from env_runner
        del env_runner
        torch.cuda.empty_cache()

        print(f"=== Completed run {run+1}/{args.n_runs} ===")
    
    # del policy to free gpu after runs
    del policy

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run diffusion policy with a specified task.")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["lift", "square", "can"],
        help="Task to run (lift, square, or can)."
    )
    parser.add_argument(
        "--use_awe",
        action="store_true",
        help="Use the AWE version of the model (task_awe_diffusion.ckpt)"
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=25,
        help="Number of parallel environments to run."
    )
    parser.add_argument(
        "--n_test_vis",
        type=int,
        default=0,
        help="Number of test episodes to visualize."
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=25,
        help="Number of test episodes to run - one for each env i.e., 20 envs means this should be 20 as well."
    )
    parser.add_argument(
        "--kp",
        type=int,
        default=150,
        help="KP control gain value."
    )
    parser.add_argument(
        "--kp_limits",
        type=int,
        nargs=2,
        default=[0, 300],
        help="KP control limits as two values (e.g., '--kp_limits 0 300')"
    )
    parser.add_argument(
        "--damping_limits",
        type=int,
        nargs=2,
        default=[0, 10],
        help="Damping limits as two values (e.g., '--damping_limits 0 10')"
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help="Number of times to repeat the experiment"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Base output directory for results"
    )

    args = parser.parse_args()
    
    # Call main function with all arguments
    main(args)