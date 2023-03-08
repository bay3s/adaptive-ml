
import time
import cloudpickle


class ExperimentStats:

  def __init__(self, total_epochs: int, total_iterations: int, total_env_steps: int, last_episode: list = None):
    """
    Stats for an experiment.

    Args:
      total_epochs (int): Total epoches.
      total_iterations (int): Total Iterations.
      total_env_steps (int): Total environment steps collected.
      last_episode (list[dict]): Last sampled episodes.
    """
    self.total_epochs = total_epochs
    self.total_iterations = total_iterations
    self.total_env_steps = total_env_steps
    self.last_episode = last_episode
    pass

