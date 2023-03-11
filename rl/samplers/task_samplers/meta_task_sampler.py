from .base_task_sampler import BaseTaskSampler
from rl.samplers.update_handlers.meta_task_update_handler import MetaTaskUpdateHandler


class MetaTaskSampler(BaseTaskSampler):

  def __init__(self, env_constructor, *, env = None, wrapper = None):
    """
    MetaTaskSampler where the environment can sample "task objects".

    Args:
      env_constructor (callable): Type of the environment.
      env (GymMetaEnv): Instance of env_constructor to sample.
      wrapper (Callable[garage.Environment, garage.Environment] or None): Wrapper function to apply to environment.
    """
    self._env_constructor = env_constructor
    self._env = env or env_constructor()
    self._wrapper = wrapper

  @property
  def n_tasks(self):
    """
    int or None: The number of tasks if known and finite.
    """
    return getattr(self._env, 'num_tasks', None)

  def sample(self, n_tasks: int, with_replacement: bool = False):
    """
    Sample a list of environment updates.

    Args:
      n_tasks (int): Number of updates to sample.
      with_replacement (bool): Whether tasks can repeat when sampled.

    Returns:
      list[MetaTaskUpdateHandler]: Batch of sampled environment updates, which, when invoked on environments, will configure them
      with new tasks.
    """
    return [
      MetaTaskUpdateHandler(self._env_constructor, task, self._wrapper)
      for task in self._env.sample_tasks(n_tasks)
    ]
