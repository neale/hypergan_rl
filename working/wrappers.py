import numpy as np
import torch
import inspect
import sys
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym.spaces import Box
from gym import Env

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
        action = np.clip(action, -1., 1.)
        lb, ub = self.unwrapped.action_space.low, self.unwrapped.action_space.high
        scaled_action = lb + (action + 1.0) * 0.5 * (ub - lb)
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


class Serializable(object):
    def __init__(self, *args, **kwargs):
        self.__args = args
        self.__kwargs = kwargs

    def quick_init(self, locals_):
        if getattr(self, "_serializable_initialized", False):
            return
        if sys.version_info >= (3, 0):
            spec = inspect.getfullargspec(self.__init__)
            # Exclude the first "self" parameter
            if spec.varkw:
                kwargs = locals_[spec.varkw].copy()
            else:
                kwargs = dict()
            if spec.kwonlyargs:
                for key in spec.kwonlyargs:
                    kwargs[key] = locals_[key]
        else:
            spec = inspect.getargspec(self.__init__)
            if spec.keywords:
                kwargs = locals_[spec.keywords]
            else:
                kwargs = dict()
        if spec.varargs:
            varargs = locals_[spec.varargs]
        else:
            varargs = tuple()
        in_order_args = [locals_[arg] for arg in spec.args][1:]
        self.__args = tuple(in_order_args) + varargs
        self.__kwargs = kwargs
        setattr(self, "_serializable_initialized", True)

    def __getstate__(self):
        return {"__args": self.__args, "__kwargs": self.__kwargs}

    def __setstate__(self, d):
        # convert all __args to keyword-based arguments
        if sys.version_info >= (3, 0):
            spec = inspect.getfullargspec(self.__init__)
        else:
            spec = inspect.getargspec(self.__init__)
        in_order_args = spec.args[1:]
        out = type(self)(**dict(zip(in_order_args, d["__args"]), **d["__kwargs"]))
        self.__dict__.update(out.__dict__)

    @classmethod
    def clone(cls, obj, **kwargs):
        assert isinstance(obj, Serializable)
        d = obj.__getstate__()
        d["__kwargs"] = dict(d["__kwargs"], **kwargs)
        out = type(obj).__new__(type(obj))
        out.__setstate__(d)
        return out

class ProxyEnv(Serializable, Env):
    def __init__(self, wrapped_env):
        Serializable.quick_init(self, locals())
        self._wrapped_env = wrapped_env
        self.action_space = self._wrapped_env.action_space
        self.observation_space = self._wrapped_env.observation_space

    @property
    def wrapped_env(self):
        return self._wrapped_env

    def reset(self, **kwargs):
        return self._wrapped_env.reset(**kwargs)

    def step(self, action):
        return self._wrapped_env.step(action)

    def render(self, *args, **kwargs):
        return self._wrapped_env.render(*args, **kwargs)

    def log_diagnostics(self, paths, *args, **kwargs):
        if hasattr(self._wrapped_env, 'log_diagnostics'):
            self._wrapped_env.log_diagnostics(paths, *args, **kwargs)

    @property
    def horizon(self):
        return self._wrapped_env.horizon

    def terminate(self):
        if hasattr(self.wrapped_env, "terminate"):
            self.wrapped_env.terminate()


class NormalizedBoxEnv(ProxyEnv, Serializable):
    """
    Normalize action to in [-1, 1].
    Optionally normalize observations and scale reward.
    """
    def __init__(
            self,
            env,
            reward_scale=1.,
            obs_mean=None,
            obs_std=None,
    ):
        # self._wrapped_env needs to be called first because
        # Serializable.quick_init calls getattr, on this class. And the
        # implementation of getattr (see below) calls self._wrapped_env.
        # Without setting this first, the call to self._wrapped_env would call
        # getattr again (since it's not set yet) and therefore loop forever.
        self._wrapped_env = env
        # Or else serialization gets delegated to the wrapped_env. Serialize
        # this env separately from the wrapped_env.
        self._serializable_initialized = False
        Serializable.quick_init(self, locals())
        ProxyEnv.__init__(self, env)
        self._should_normalize = not (obs_mean is None and obs_std is None)
        if self._should_normalize:
            if obs_mean is None:
                obs_mean = np.zeros_like(env.observation_space.low)
            else:
                obs_mean = np.array(obs_mean)
            if obs_std is None:
                obs_std = np.ones_like(env.observation_space.low)
            else:
                obs_std = np.array(obs_std)
        self._reward_scale = reward_scale
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        ub = np.ones(self._wrapped_env.action_space.shape)
        self.action_space = Box(-1 * ub, ub)

    def estimate_obs_stats(self, obs_batch, override_values=False):
        if self._obs_mean is not None and not override_values:
            raise Exception("Observation mean and std already set. To "
                            "override, set override_values to True.")
        self._obs_mean = np.mean(obs_batch, axis=0)
        self._obs_std = np.std(obs_batch, axis=0)

    def _apply_normalize_obs(self, obs):
        return (obs - self._obs_mean) / (self._obs_std + 1e-8)

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        # Add these explicitly in case they were modified
        d["_obs_mean"] = self._obs_mean
        d["_obs_std"] = self._obs_std
        d["_reward_scale"] = self._reward_scale
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self._obs_mean = d["_obs_mean"]
        self._obs_std = d["_obs_std"]
        self._reward_scale = d["_reward_scale"]

    def step(self, action):
        lb = self._wrapped_env.action_space.low
        ub = self._wrapped_env.action_space.high
        scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)

        wrapped_step = self._wrapped_env.step(scaled_action)
        next_obs, reward, done, info = wrapped_step
        if self._should_normalize:
            next_obs = self._apply_normalize_obs(next_obs)
        return next_obs, reward * self._reward_scale, done, info

    def __str__(self):
        return "Normalized: %s" % self._wrapped_env

    def log_diagnostics(self, paths, **kwargs):
        if hasattr(self._wrapped_env, "log_diagnostics"):
            return self._wrapped_env.log_diagnostics(paths, **kwargs)
        else:
            return None

    def __getattr__(self, attrname):
        return getattr(self._wrapped_env, attrname)


