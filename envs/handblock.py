import numpy as np

from gym import utils
from gym.envs.mujoco import mujoco_env

import os

from mujoco_py.generated import const

import gym
from gym.envs.robotics import robot_env
from scipy.spatial.transform import Rotation as R

rotations = np.zeros((8, 8, 8), dtype=float)
def update_rotation_buffer(state):
    #[ x y z ] euler angles
    global rotations
    object_qpos = state[-7:]
    assert len(object_qpos) == 7
    quat = object_qpos[3:]
    if sum(quat) == 0:
        return 

    rotation = R.from_quat(quat).as_euler('xyz', degrees=True)
    for i in range(3):
        a = int(rotation[i]) + 180
        nbin = np.floor(a / 45)
        rotation[i] = int(nbin)

    rotation = rotation.astype(np.uint8)
    rotations[rotation[0], rotation[1], rotation[2]] = 1

def rotation_buffer(buffer):
    for state in buffer.states:
        update_rotation_buffer(state)
    n_unique = rotations.sum()
    return n_unique

class IntrinsicHandBlockEnv(gym.Wrapper):
    """
    Observation Space:
        - x torso COM velocity
        - y torso COM velocity
        - 24 joint positions
        - 24 joint velocities
        - 7 object state (3 dim xyz, 4 dim quat)
    """
    def __init__(self):
        env = gym.make('HandManipulateBlock-v0')
        super(IntrinsicHandBlockEnv, self).__init__(env)
    
    def reset(self):
        self.env.reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self.env.unwrapped._reset_sim()
        self.goal = self.env.unwrapped._sample_goal().copy()
        obs = self.env.unwrapped._get_obs()
        obs = obs['observation']
        return obs

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.env.unwrapped._set_action(action)
        self.env.unwrapped.sim.step()
        self.env.unwrapped._step_callback()
        obs = self.env.unwrapped._get_obs()
        obs = obs['observation'] # remove fluff about the goal pose
        return obs, None, False, {}

    @property
    def tasks(self):
        t = dict()
        return t
