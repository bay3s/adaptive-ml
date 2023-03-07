from typing import Tuple, Dict
from abc import ABC, abstractmethod

import torch
import numpy as np
import akro

from rl.structs import (
  EnvSpec
)


class BasePolicy(torch.nn.Module, ABC):

  def __init__(self, env_spec: EnvSpec, unique_id: str):
    """
    Base class for policies, provides abstract methods that should be implemented for a policy.

    Args:
      env_spec (EnvSpec): Specifications for the environment.
      unique_id (str): Unique identifier for the policy.
    """
    super().__init__()
    self._env_spec = env_spec
    self._unique_id = unique_id
    pass

  @abstractmethod
  def get_action(self, observation: torch.Tensor) -> Tuple[np.ndarray]:
    """
    Get action sampled from the policy.

    Args:
      observation (torch.Tensor): Observation from the environment.

    Returns:
      Tuple[np.ndarray, dict[str,np.ndarray]]: Actions and extra agent infos.
    """
    raise NotImplementedError

  @abstractmethod
  def get_actions(self, observations: torch.Tensor) -> Tuple:
    """
    Get actions given observations.

    Args:
      observations (torch.Tensor): Observations from the environment.

    Returns:
      Tuple[np.ndarray, dict[str,np.ndarray]]: Actions and extra agent infos.
    """
    raise NotImplementedError

  def get_param_values(self) -> Dict:
    """
    Get the parameters for the policy.

    Returns:
      dict: The parameters (in the form of the state dictionary).
    """
    return self.state_dict()

  def set_param_values(self, state_dict: Dict) -> None:
    """
    Set the parameters to the policy.

    Args:
      state_dict (dict): State dictionary.
    """
    self.load_state_dict(state_dict)
    pass

  def reset(self, do_resets: np.ndarray = None) -> None:
    """
    Reset the policy.

    This is applicable only to recurrent policies. do_resets is an array of boolean indicating
    which internal states to be reset.

    The length of do_resets should be equal to the length of inputs, i.e. batch size.

    Args:
      do_resets (numpy.ndarray): Bool array indicating which states to be reset.
    """
    pass

  @property
  def unique_id(self) -> str:
    """
    Name of policy.

    Returns:
      str: Name of policy
    """
    return self._unique_id

  @property
  def env_spec(self) -> EnvSpec:
    """
    Policy environment specification.

    Returns:
      garage.EnvSpec: Environment specification.
    """
    return self._env_spec

  @property
  def observation_space(self):
    """
    Observation space.

    Returns:
      akro.Space: The observation space of the environment.
    """
    return self.env_spec.observation_space

  @property
  def action_space(self) -> akro.Space:
      """
      Action space.

      Returns:
        akro.Space: The action space of the environment.
      """
      return self.env_spec.action_space
