from abc import ABC, abstractmethod

import akro
import numpy as np
import torch

from .base_policy import BasePolicy
from typing import Tuple

from rl.utils.functions.preprocessing_functions import np_to_torch, list_to_tensor


class StochasticPolicy(BasePolicy, ABC):

  def get_action(self, observation: np.ndarray) -> Tuple:
    """
    Get a single action given an observation.

    Returns the following two in a tuple,
    - the predicted action whose shape is equal to the shape of the action space.
    - dictionary containing the mean of the distribution and standard deviation of the logarithmic values of the
      distribution.

    Args:
      observation (np.ndarray): Observation from the environment.

    Returns:
      Tuple[np.ndarray, dct]
    """
    if not isinstance(observation, np.ndarray) and not isinstance(observation, torch.Tensor):
      observation = self._env_spec.observation_space.flatten(observation)
    elif isinstance(observation, np.ndarray) and len(observation.shape) > 1:
      observation = self._env_spec.observation_space.flatten(observation)

    elif isinstance(observation, torch.Tensor) and len(observation.shape) > 1:
      observation = torch.flatten(observation)

    with torch.no_grad():
      if isinstance(observation, np.ndarray):
        observation = np_to_torch(observation)

      if not isinstance(observation, torch.Tensor):
        observation = list_to_tensor(observation)

      observation = observation.unsqueeze(0)
      action, agent_infos = self.get_actions(observation)

      return action[0], {k: v[0] for k, v in agent_infos.items()}

  def get_actions(self, observations: np.ndarray) -> Tuple:
    """
    Get actions given multiple observations.

    Returns the following two in a tuple,
    - the predicted action whose shape is equal to the shape of the action space.
    - dictionary containing the mean of the distribution and standard deviation of the logarithmic values of the
      distribution.

    Args:
      observations (np.ndarray): Observations from the environment.

    Returns:
      Tuple[np.ndarray, dict]
    """
    if not isinstance(observations[0], np.ndarray) and not isinstance(observations[0], torch.Tensor):
      observations = self._env_spec.observation_space.flatten_n(observations)

    if isinstance(observations, list):
      if isinstance(observations[0], np.ndarray):
        observations = np.stack(observations)
      elif isinstance(observations[0], torch.Tensor):
        observations = torch.stack(observations)

    if isinstance(observations[0], np.ndarray) and len(observations[0].shape) > 1:
      observations = self._env_spec.observation_space.flatten_n(observations)

    elif isinstance(observations[0], torch.Tensor) and len(observations[0].shape) > 1:
      observations = torch.flatten(observations, start_dim = 1)

    with torch.no_grad():
      if isinstance(observations, np.ndarray):
        observations = np_to_torch(observations)

      if not isinstance(observations, torch.Tensor):
        observations = list_to_tensor(observations)

      if isinstance(self._env_spec.observation_space, akro.Image):
        observations /= 255.0  # scale image

      dist, info = self.forward(observations)

      return dist.sample().cpu().numpy(), {
        k: v.detach().cpu().numpy()
        for (k, v) in info.items()
      }

    pass

  @abstractmethod
  def forward(self, observations: torch.Tensor) -> Tuple:
    """
    Compute the action distributions from the observations.

    Args:
      observations (torch.Tensor): Batch of observations as tensors on the default device.

    Returns:
      Tuple[torch.distributions.Distribution, dict]: Batch distribution of actions and additional info about the agent.
    """
    raise NotImplementedError
