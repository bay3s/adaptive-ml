from abc import ABC, abstractmethod


class BaseWorker(ABC):

  def __init__(self, *, seed: int, max_episode_length: int, worker_number: int):
    """
    Initialize a worker.

    Args:
      seed (int): The seed to use to intialize random number generators.
      max_episode_length (int or float): The maximum length of episodes which will be sampled.
      worker_number (int): The number of the worker this update is occurring in.
    """
    self._seed = seed
    self._max_episode_length = max_episode_length
    self._worker_number = worker_number
    pass

  @abstractmethod
  def update_agent(self, agent_update):
    """
    Update the worker's agent, using agent_update.

    Args:
      agent_update (object): An agent update.
    """
    raise NotImplementedError

  @abstractmethod
  def update_env(self, env_update):
    """
    Update the worker's env, using env_update.

    Args:
      env_update (object): An environment update.
    """
    raise NotImplementedError

  @abstractmethod
  def rollout(self):
    """
    Sample a single episode of the agent in the environment.

    Returns:
      EpisodeBatch: Batch of sampled episodes. May be truncated if max_episode_length is set.
    """
    raise NotImplementedError

  @abstractmethod
  def start_episode(self):
    """
    Begin a new episode.
    """
    raise NotImplementedError

  @abstractmethod
  def step_episode(self):
    """
    Take a single time-step in the current episode.

    Returns:
      True iff the episode is done.
    """
    raise NotImplementedError

  @abstractmethod
  def collect_episode(self):
    """
    Collect the current episode, clearing the internal buffer.

    Returns:
      EpisodeBatch: Batch of sampled episodes. May be truncated if the episodes haven't completed yet.
    """
    raise NotImplementedError

  @abstractmethod
  def shutdown(self):
    """
    Shutdown the worker.
    """
    raise NotImplementedError

  def __getstate__(self):
    """
    Refuse to be pickled.

    Raises:
      ValueError: Always raised, since pickling Workers is not supported.
    """
    raise ValueError('Workers are not pickleable, please pickle the WorkerFactory instead.')
