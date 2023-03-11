from abc import ABC, abstractmethod


class BaseTaskSampler(ABC):

  @abstractmethod
  def sample(self, n_tasks, with_replacement: bool = False):
    """
    Sample a list of environment updates.

    Args:
      n_tasks (int): Number of updates to sample.
      with_replacement (bool): Whether tasks can repeat when sampled.

    Returns:
      list[EnvUpdateHandler]: Batch of sampled environment updates, which, when invoked on environments, will configure
        them with new tasks.
    """
    pass

  @property
  def n_tasks(self, num_tasks: int):
    """
    int: The number of tasks if known and finite.
    """
    return None
