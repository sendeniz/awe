import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import (
    VideoRecordingWrapper,
    VideoRecorder,
)
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
import json 

def save_to_json(list_of_arrays, name):  # Changed parameter to 'name' for consistency
    """
    Save a list of numpy arrays to JSON, preserving their separation.
    Each array will be stored as a separate entry in the JSON list.
    """
    # Convert all arrays to nested lists (JSON-compatible)
    json_data = [arr.tolist() for arr in list_of_arrays]
    
    with open(f"{name}.json", "w") as f:
        json.dump(json_data, f)

def create_env(env_meta, shape_meta, enable_render=True):

    env_meta["env_kwargs"]["camera_heights"] = 256
    env_meta["env_kwargs"]["camera_widths"] = 256

    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta["obs"].items():
        modality_mapping[attr.get("type", "low_dim")].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=enable_render,
    )
   
    return env


class RobomimicImageRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(
        self,
        output_dir,
        dataset_path,
        shape_meta: dict,
        n_train=10,
        n_train_vis=3,
        train_start_idx=0,
        n_test=22,
        n_test_vis=6,
        test_start_seed=10000,
        max_steps=400,
        n_obs_steps=2,
        n_action_steps=8,
        render_obs_key="agentview_image",
        fps=10,
        crf=22,
        past_action=False,
        abs_action=False,
        tqdm_interval_sec=5.0,
        n_envs=None,
        multiplier=10,
    ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)



        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        # disable object state observation
        env_meta["env_kwargs"]["use_object_obs"] = False
        
        env_meta["env_kwargs"]["controller_configs"]["kp"] = 1000 # 150 #  1000 
        env_meta["env_kwargs"]["controller_configs"]["kp_limits"] = [0, 2000] # [0, 300] #   [0, 2000] #
        env_meta["env_kwargs"]["controller_configs"]["damping_limits"] = [0, 60] # [0, 10] #  [0, 10]  #  [0, 60]

        rotation_transformer = None
        if abs_action:
            env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False
            rotation_transformer = RotationTransformer("axis_angle", "rotation_6d")
        
        # add linear interpolators for pos and ori
        if multiplier > 1:
            env_meta["env_kwargs"]["controller_configs"]["interpolation"] = "linear"
            env_meta["env_kwargs"]["controller_configs"]["multiplier"] = multiplier
        print("env_meta", env_meta)

        def env_fn():
            robomimic_env = create_env(env_meta=env_meta, shape_meta=shape_meta)
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )

        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robomimic_env = create_env(
                env_meta=env_meta, shape_meta=shape_meta, enable_render=False
            )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key,
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec="h264",
                        input_pix_fmt="rgb24",
                        crf=crf,
                        thread_type="FRAME",
                        thread_count=1,
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render,
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps,
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, "r") as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f"data/demo_{train_idx}/states"][0]

                def init_fn(env, init_state=init_state, enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            "media", wv.util.generate_id() + ".mp4"
                        )
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicImageWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append("train/")
                env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        "media", wv.util.generate_id() + ".mp4"
                    )
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicImageWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append("test/")
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
        # env = SyncVectorEnv(env_fns)

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each("run_dill_function", args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta["env_name"]
            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec,
            )

            done = False
            reward_lst = []
            # OSC controller inputs to store end effector 3D pos and how far grippen is open/closed
            controller_inputs_lst = []
            # list to store end effector position in sim
            end_effector_positions_lst = []
            avg_euclid_error_lst = []

            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict["past_action"] = past_action[
                        :, -(self.n_obs_steps - 1) :
                    ].astype(np.float32)

                # device transfer
                obs_dict = dict_apply(
                    np_obs_dict, lambda x: torch.from_numpy(x).to(device=device)
                )

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(
                    action_dict, lambda x: x.detach().to("cpu").numpy()
                )

                action = np_action_dict["action"]
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")

                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)
                

                # fetch last time step from controller inputs not env
                # fetch last time-step from controller inputs acrooss number of envs all envs cooresponds to first idex with value ":"
                last_timesteps = env_action[:, -1, :]
                # fetch 3D coordinates and gripper state
                coordinates = last_timesteps[:, :3] 
                gripper_states = last_timesteps[:, -1]
                 # Shape from (2, ) to (2, 1) for concat operation

                controller_input = np.hstack((coordinates, gripper_states[:, np.newaxis]))
                controller_inputs_lst.append(controller_input)

                obs, reward, done, info = env.step(env_action)

                # append reward
                reward_lst.append(reward)
                
                # Fetch joint positions
                # joint_positions = env.call("get_joint_3d_positions")

                end_effector_position = np.array(env.call("get_end_effector_3d_position"))

                # Fetch joint state from sim
                gripper_state_sim = np.array(env.call("get_gripper_state"))
            
                # controller inputs from env
                controller_input_env = np.hstack((end_effector_position, gripper_state_sim[:, np.newaxis]))

                end_effector_positions_lst.append(controller_input_env)
                
                # compute eucledian error between controller inputs and actual eef pos from env
                euclid_error = np.linalg.norm(coordinates - end_effector_position, axis=1)

                # average to get avg obtain error over all environments
                avg_euclid_error = np.mean(euclid_error)
                avg_euclid_error_lst.append(avg_euclid_error)

                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # print metrics: mean reward, mean euclid. error and successrate
            # average metrics over each time step
            avg_reward = sum(reward_lst) / len(reward_lst)
            # average mean reward over number of envs here, bcs if individual env has zero reward it failed
            # can be used to compute rate of sucess 
            mean_reward = np.array([np.mean(avg_reward)]) 
            # compute rate of success
            n_successes =  np.count_nonzero(avg_reward > 0)
            successes_rate = n_successes / len(avg_reward) * 100
            mean_euclid_error =  sum(avg_euclid_error_lst) / len(avg_euclid_error_lst)
            print(f"Mean reward: {mean_reward}, Mean 3D error: {mean_euclid_error}, Success rate {successes_rate}:")
            
            save_to_json(mean_reward, name = "results/square/mean_reward_kp1K_awe")
            save_to_json(mean_euclid_error, name = "results/square/mean_euclid_error_kp1K_awe")

            # save 3D positions and gripper state lists to a csv
            #print("controller_inputs_lst.shape:", controller_inputs_lst.shape)
            #print("end_effector_positions_lst.shape:", end_effector_positions_lst.shape)
            #print("controller_inputs_lst:", controller_inputs_lst)
            save_to_json(controller_inputs_lst, name = "results/square/osc_inputs_kp1K_awe")
            save_to_json(end_effector_positions_lst, name = "results/square/eef_pos_kp1K_awe")

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call("get_attr", "reward")[
                this_local_slice
            ]
        # clear out video buffer
        _ = env.reset()

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        for i in range(len(self.env_fns)):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix + f"sim_max_reward_{seed}"] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix + f"sim_video_{seed}"] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix + "mean_score"
            value = np.mean(value)
            log_data[name] = value

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1, 2, 10)

        d_rot = action.shape[-1] - 4
        pos = action[..., :3]
        rot = action[..., 3 : 3 + d_rot]
        gripper = action[..., [-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([pos, rot, gripper], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction