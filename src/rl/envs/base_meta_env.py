from gym.core import Env
import numpy as np


class BaseMetaEnv(Env):

  def sample_tasks(self, n_tasks):
    """
    Samples task of the meta-environment.

    Args:
      n_tasks (int): Number of different meta-tasks to sample.

    Returns:

    """
    raise NotImplementedError()

  def set_task(self, task):
    """
    Sets the specificied task to the current enviroment.

    Args:
      task (): Task from the meta-learning environment.

    Returns:

    """
    raise NotImplementedError()

  def get_task(self):
    """
    Gets the task that the agent is performing in the current environment.

    Returns:
      task: Task of the meta-learning environment.
    """
    raise NotImplementedError()

  def log_diagnostics(self, paths, prefix):
    """
    Logs env-specific diagnostic information.

    Args:
      paths (list): List of all paths collected with the env during this iteration.
      prefix (str): Prefix for the logger.

    Returns:

    """
    pass
