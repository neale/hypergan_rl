import numpy as np
import torch

import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder


def toTensor(array, dtype=None):
    if dtype is None:
        dtype = torch.get_default_dtype()
    if isinstance(array, (list, tuple)):
        array = torch.cat([toTensor(x) for x in array], dim=0)
    if isinstance(array, int):
        array = float(array)
    if isinstance(array, float):
        array = [array, ]
    if isinstance(array, list):
        array = np.array(array)
    if isinstance(array, (np.ndarray, np.bool_, np.float32, np.float64, np.int32, np.int64)):
        if array.dtype == np.bool_:
            array = array.astype(np.uint8)
        array = torch.tensor(array, dtype=dtype)
        # array = array.unsqueeze(0)
    while len(array.shape) < 2:
        array = array.unsqueeze(0)
    return array


class BoundedActionsEnv(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=self.unwrapped.action_space.shape)

    def step(self, action):
        # action = np.clip(action, -1., 1.)
        lb, ub = self.unwrapped.action_space.low, self.unwrapped.action_space.high
        scaled_action = lb + (action + 1.0) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)
        observation, reward, done, info = self.env.step(scaled_action)
        return observation, reward, done, info


class RecordedEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, filename=''):
        if hasattr(self, 'recorder'):
            self.recorder.capture_frame()
            self.recorder.close()
        self.recorder = VideoRecorder(self.env, path=filename)
        return self.env.reset()

    def step(self, action):
        self.recorder.capture_frame()
        return self.env.step(action)

    def close(self):
        if hasattr(self, 'recorder'):
            self.recorder.capture_frame()
            self.recorder.close()
            del self.recorder
        return self.env.close()


class NoisyEnv(gym.Wrapper):
    def __init__(self, env, stdev):
        self.stdev = stdev
        super().__init__(env)

    def noisify(self, state):
        state += np.random.normal(scale=self.stdev, size=state.size)
        return state

    def reset(self, filename=''):
        state = self.env.reset()
        return self.noisify(state)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return self.noisify(state), reward, done, info


class TorchEnv(gym.Wrapper):
    def __init__(self, env, device=torch.device('cuda')):
        super().__init__(env)
        self.device = device

    def _convert_state(self, state):
        if isinstance(state, (float, int)):
            state = toTensor(state).to(self.device)
        if isinstance(state, np.ndarray):
            state = toTensor(state).to(self.device)
        return state

    def _convert_action(self, action):
        if isinstance(action, torch.Tensor):
            action = action.view(-1).detach().data.cpu().numpy()
        return action

    def reset(self, *args, **kwargs):
        state = self.env.reset(*args, **kwargs)
        state = self._convert_state(state)
        return state

    def step(self, action):
        action = self._convert_action(action)
        state, reward, done, info = self.env.step(action)
        state = self._convert_state(state)
        reward = self._convert_state(reward).squeeze(0)
        return state, reward, done, info

