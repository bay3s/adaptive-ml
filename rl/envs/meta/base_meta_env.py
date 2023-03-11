from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from gym import Env


class BaseMetaEnv(Env, ABC):

  def __init__(self):
    """
    Wrapper around OpenAI gym environments, interface for meta learning.
    """
    super().__init__()

    self.np_random = np.random
    self._np_random = np.random
    pass

  @abstractmethod
  def sample_tasks(self, n_tasks: int) -> np.ndarray:
    """
    Samples task of the meta-environment

    Args:
        n_tasks (int) : number of different meta-tasks needed

    Returns:
        tasks (list) : an (n_tasks) length list of tasks
    """
    raise NotImplementedError

  @abstractmethod
  def set_task(self, task) -> None:
    """
    Sets the specified task to the current environment

    Args:
        task: task of the meta-learning environment
    """
    raise NotImplementedError

  @abstractmethod
  def _get_obs(self) -> np.ndarray:
    """
    Returns the observation from the current state of the environment

    Args:
      np.ndarray
    """
    raise NotImplementedError

  @abstractmethod
  def get_task(self) -> Union[float, np.ndarray]:
    """
    Gets the task that the agent is performing in the current environment

    Returns:
      np.ndarray
    """
    raise NotImplementedError
