import warnings
from typing import Callable

from .base_env_update_handler import BaseEnvUpdateHandler


class MetaTaskUpdateHandler(BaseEnvUpdateHandler):

  def __init__(self, env_type: type, task: object, wrapper_constructor: Callable):
    """
    Calls set_task with the provided task.

    Args:
      env_type (type): Type of environment.
      task (object): Opaque task type.
      wrapper_constructor (Callable[Env, Env] or None): Callable that wraps constructed environments.
    """
    if not isinstance(env_type, type):
      raise ValueError(f'env_type should be a type, not {type(env_type)!r}')

    self._env_type = env_type
    self._task = task
    self._wrapper_cons = wrapper_constructor
    pass

  def _make_env(self):
    """
    Construct the environment, wrapping if necessary.

    Returns:
      GymEnv: The (possibly wrapped) environment.
    """
    env = self._env_type()
    env.set_task(self._task)

    if self._wrapper_cons is not None:
      env = self._wrapper_cons(env, self._task)

    return env

  def __call__(self, old_env=None):
    """
    Update an environment.

    Args:
      old_env (Environment or None): Previous environment.

    Returns:
      Environment
    """
    if old_env is None:
      return self._make_env()
    elif type(getattr(old_env, 'unwrapped', old_env)) != self._env_type:
      warnings.warn('Closing an environment, this may indicate a very slow TaskSampler setup.')
      old_env.close()
      return self._make_env()

    old_env.set_task(self._task)
    return old_env
