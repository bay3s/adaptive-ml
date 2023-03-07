from typing import Callable
from abc import ABC, abstractmethod


class EnvUpdateHandler(ABC):

  @abstractmethod
  def __call__(self, old_env=None):
    """
    A callable that "updates" an environment.
    """
    raise NotImplementedError


class NewEnvHandler(EnvUpdateHandler):

  def __init__(self, env_constructor: Callable):
    """
    Takes a constructor for the new environment and returns a new instance on __call__.

    Args:
      env_constructor ():
    """
    self._env_constructor = env_constructor

  def __call__(self, old_env = None):
    """
    Update an environment.

    Args:
      old_env (Environment or None): Previous environment.

    Returns:
      Environment: The new, updated environment.
    """
    if old_env:
      old_env.close()

    return self._env_constructor()
