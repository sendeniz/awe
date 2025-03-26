from typing import List, Optional
from matplotlib.pyplot import fill
import numpy as np
import gym
from gym import spaces
from omegaconf import OmegaConf
from robomimic.envs.env_robosuite import EnvRobosuite
import cv2 
import mujoco


def img_resize(img, hw=(84, 84)):
    # from 3, H, W to H, W, 3
    img = img.transpose(1, 2, 0)
    # Resize image to (H, W, 3) to Resize H, Resize W, 3 
    img = cv2.resize(img, (84, 84))
    # H, W, 3 back to 3, H, W
    img = img.transpose(2, 0, 1)

    return img

def normalize(value, in_min, in_max, out_min, out_max):
    return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

class RobomimicImageWrapper(gym.Env):
    def __init__(self, 
        env: EnvRobosuite,
        shape_meta: dict,
        init_state: Optional[np.ndarray]=None,
        render_obs_key='agentview_image',
        ):

        self.env = env
        self.render_obs_key = render_obs_key
        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False
        
        # setup spaces
        action_shape = shape_meta['action']['shape']
        action_space = spaces.Box(
            low=-1,
            high=1,
            shape=action_shape,
            dtype=np.float32
        )
        self.action_space = action_space

        observation_space = spaces.Dict()
        for key, value in shape_meta['obs'].items():
            shape = value['shape']
            min_value, max_value = -1, 1
            if key.endswith('image'):
                min_value, max_value = 0, 1
            elif key.endswith('quat'):
                min_value, max_value = -1, 1
            elif key.endswith('qpos'):
                min_value, max_value = -1, 1
            elif key.endswith('pos'):
                # better range?
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")
            
            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space


    
    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()
        
        self.render_cache = raw_obs[self.render_obs_key]
        
        if self.render_obs_key in raw_obs:
            img = raw_obs[self.render_obs_key]
            reder_obs_img = img_resize(img, hw=(84, 84))

        img = raw_obs["robot0_eye_in_hand_image"]
        eye_in_hand_obs_img = img_resize(img, hw=(84, 84))
       

        #print(f"Processed image shape: {reder_obs_img.shape}")  # Debugging print
        #print(f"Processed image shape: {eye_in_hand_obs_img.shape}")  # Debugging print
        
        # Store resized image
        raw_obs[self.render_obs_key] = reder_obs_img 
        raw_obs["robot0_eye_in_hand_image"] = eye_in_hand_obs_img
        
        obs = dict()
        for key in self.observation_space.keys():
            obs[key] = raw_obs[key]

        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self):
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state
            # to be compatible with gym
            raw_obs = self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                raw_obs = self.env.reset_to({'states': self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                state = self.env.get_state()['states']
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            raw_obs = self.env.reset()

        # return obs
        obs = self.get_observation(raw_obs)
        return obs
    
    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)
        return obs, reward, done, info
    
    def render(self, mode='rgb_array'):
        if self.render_cache is None:
            raise RuntimeError('Must run reset or step before render.')
        img = np.moveaxis(self.render_cache, 0, -1)
        img = (img * 255).astype(np.uint8)
        return img
    
    # Add this method to access joint positions
    def get_joint_positions(self):
        """
        Returns the joint positions (qpos) from the underlying Robosuite environment.
        """
        return self.env.env.sim.data.qpos.copy()
    
    def get_joint_3d_positions(self):
        """
        Returns the 3D positions of the robot's joints in world coordinates.
        """
        model = self.env.env.sim.model
        data = self.env.env.sim.data
        
        joint_positions_3d = []
        
        for joint_id in range(model.njnt):  # Iterate over joints
            joint_name = joint_name = model.joint_id2name(joint_id)
            #print(f"Joint nanmes: {joint_name}")
            if joint_name:  # Ensure the joint has a name
                body_id = model.jnt_bodyid[joint_id]  # Get the body attached to the joint
                joint_positions_3d.append(data.body_xpos[body_id])  # Get the world position of the body
        
        return np.array(joint_positions_3d)
    
    def get_end_effector_3d_position(self):
        """
        Returns the 3D position of the robot's end-effector in world coordinates.
        """
        data = self.env.env.sim.data
        
        # Use the identified end-effector site name
        end_effector_site_name = "gripper0_grip_site"
        
        # Get the site ID for the end-effector
        site_id = self.env.env.sim.model.site_name2id(end_effector_site_name)
        
        # Get the 3D position of the end-effector site
        end_effector_pos = data.site_xpos[site_id]
        return end_effector_pos

    def get_gripper_state(self):
        # Get all joint positions
        joint_positions = self.env.env.sim.data.qpos.copy()

        gripper_joint_names = ["gripper0_finger_joint1", "gripper0_finger_joint2"]
        gripper_joint_indices = [self.env.env.sim.model.joint_name2id(name) for name in gripper_joint_names]

        # Extract the joint positions for the gripper fingers
        gripper_joint_positions = joint_positions[gripper_joint_indices]

        input_min = -1
        input_max = 1
        output_min = -1  # Gripper state range: -1 (closed) to 1 (open)
        output_max = 1

        normalized_joint1 = normalize(gripper_joint_positions[0], input_min, input_max, output_min, output_max)
        normalized_joint2 = normalize(gripper_joint_positions[1], input_min, input_max, output_min, output_max)

        # Compute the gripper state as the average of the two normalized joint positions
        gripper_state = -((normalized_joint1 + normalized_joint2) / 2) - 1

        #print("Gripper State:", gripper_state)
        return gripper_state
    


def test():
    import os
    from omegaconf import OmegaConf
    cfg_path = os.path.expanduser('~/dev/diffusion_policy/diffusion_policy/config/task/lift_image.yaml')
    cfg = OmegaConf.load(cfg_path)
    shape_meta = cfg['shape_meta']


    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    from matplotlib import pyplot as plt

    dataset_path = os.path.expanduser('~/dev/diffusion_policy/data/robomimic/datasets/square/ph/image.hdf5')
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=False,
        use_image_obs=True, 
    )

    wrapper = RobomimicImageWrapper(
        env=env,
        shape_meta=shape_meta
    )
    wrapper.seed(0)
    obs = wrapper.reset()
    img = wrapper.render()
    plt.imshow(img)


    # states = list()
    # for _ in range(2):
    #     wrapper.seed(0)
    #     wrapper.reset()
    #     states.append(wrapper.env.get_state()['states'])
    # assert np.allclose(states[0], states[1])

    # img = wrapper.render()
    # plt.imshow(img)
    # wrapper.seed()
    # states.append(wrapper.env.get_state()['states'])
