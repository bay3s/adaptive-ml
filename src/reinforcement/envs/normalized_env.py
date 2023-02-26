from typing import Any, Tuple, Union
from gym import Env
from gym.spaces import Space
import numpy as np
from gym.spaces import Box

from src.reinforcement.utils.serializable import Serializable


class NormalizedEnv(Serializable):

  def __init__(self, env: Env, scale_reward: float = 1., normalize_obs: bool = False, normalize_reward: bool = False,
               obs_alpha: float = 0.001, reward_alpha: float = 0.001, normalization_scale: float = 10.):
    """
    Initialize a normalized environment based on the Gym environment provided in the arguments.

    Args:
      env (Env): Gym environment to be normalized.
      scale_reward (float): Scaling for the rewards in the environment.
      normalize_obs (bool): Whether to normalize observations from the environment.
      normalize_reward (bool): Whether to normalize rewards in the environment as well.
      obs_alpha (float): Alpha value to be used for observations.
      reward_alpha (float): Alpha value to be used for rewards.
      normalization_scale (float): Normalization scale to be used for the environment.
    """
    Serializable.__init__(self)
    Serializable.quick_init(self, locals())

    self._scale_reward = scale_reward
    self._wrapped_env = env

    self._normalize_obs = normalize_obs
    self._normalize_reward = normalize_reward
    self._obs_alpha = obs_alpha
    self._obs_mean = np.zeros(self.observation_space.shape)
    self._obs_var = np.ones(self.observation_space.shape)
    self._reward_alpha = reward_alpha
    self._reward_mean = 0.
    self._reward_var = 1.
    self._normalization_scale = normalization_scale

  @property
  def action_space(self) -> Space:
    """
    Returns the action space of the current environment.

    Returns:
      Space
    """
    if isinstance(self._wrapped_env.action_space, Box):
      ub = np.ones(self._wrapped_env.action_space.shape) * self._normalization_scale

      return Box(-1 * ub, ub, dtype = np.float32)

    return self._wrapped_env.action_space

  def __getattr__(self, attr: str) -> Any:
    """
    If normalized env does not have the attribute then call the attribute in the wrapped_env

    Args:
      attr (str): Attribute to fetch.

    Returns:
      Any
    """
    orig_attr = self._wrapped_env.__getattribute__(attr)

    if callable(orig_attr):
      def hooked(*args, **kwargs):
        result = orig_attr(*args, **kwargs)
        return result

      return hooked
    else:
      return orig_attr

  def _update_obs_estimate(self, obs) -> None:
    """
    Updates the mean and variance estimates based on a given observation.

    Args:
      obs (np.ndarray): Observation that is made.

    Returns:
      None
    """
    o_a = self._obs_alpha
    self._obs_mean = (1 - o_a) * self._obs_mean + o_a * obs
    self._obs_var = (1 - o_a) * self._obs_var + o_a * np.square(obs - self._obs_mean)

  def _update_reward_estimate(self, reward: float) -> None:
    """
    Update reward estimate based on the given reward.

    Args:
      reward (float): Reward observed in the environment.

    Returns:
      None
    """
    r_a = self._reward_alpha
    self._reward_mean = (1 - r_a) * self._reward_mean + r_a * reward
    self._reward_var = (1 - r_a) * self._reward_var + r_a * np.square(reward - self._reward_mean)

  def _apply_normalize_obs(self, obs: np.ndarray) -> np.ndarray:
    """
    Normalize an observation.

    Args:
      obs (np.ndarray): Observation to normalize.

    Returns:
      float
    """
    self._update_obs_estimate(obs)

    return (obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)

  def _apply_normalize_reward(self, reward: float) -> float:
    """
    Given a reward, normalizes and returns the result.

    Args:
      reward (float):

    Returns:
      float
    """
    self._update_reward_estimate(reward)

    return reward / (np.sqrt(self._reward_var) + 1e-8)

  def reset(self) -> np.ndarray:
    """
    Reset the environment and return the resulting observation.

    Returns:
      np.ndarray
    """
    obs = self._wrapped_env.reset()

    if self._normalize_obs:
      return self._apply_normalize_obs(obs)

    return obs

  def __getstate__(self) -> dict:
    """
    Return the current state of the NormalizedEnv object.

    Returns:
     dict
    """
    d = Serializable.__getstate__(self)
    d['_obs_mean'] = self._obs_mean
    d['_obs_var'] = self._obs_var

    return d

  def __setstate__(self, d: dict) -> None:
    """
    Set the state of a NormalizedEnv object.

    Args:
      d (dict):

    Returns:
      None
    """
    Serializable.__setstate__(self, d)
    self._obs_mean = d['_obs_mean']
    self._obs_var = d['_obs_var']
    pass

  def step(self, action: Union[float, int]) -> Tuple:
    """
    Take a step in the environment.

    Args:
      action (Union[float, int]): Action to take in the environment.

    Returns:
      Tuple
    """
    if isinstance(self._wrapped_env.action_space, Box):
      # rescale the action
      lb, ub = self._wrapped_env.action_space.low, self._wrapped_env.action_space.high
      scaled_action = lb + (action + self._normalization_scale) * (ub - lb) / (2 * self._normalization_scale)
      scaled_action = np.clip(scaled_action, lb, ub)
    else:
      scaled_action = action

    wrapped_step = self._wrapped_env.step(scaled_action)
    next_obs, reward, done, info = wrapped_step

    if getattr(self, '_normalize_obs', False):
      next_obs = self._apply_normalize_obs(next_obs)

    if getattr(self, '_normalize_reward', False):
      reward = self._apply_normalize_reward(reward)

    return next_obs, reward * self._scale_reward, done, info
