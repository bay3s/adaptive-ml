from abc import ABC, abstractmethod
from typing import List, Union

from rl.structs import EpisodeBatch
from rl.envs.base_env import BaseEnv
from rl.networks.policies.base_policy import BasePolicy

from .workers import WorkerFactory


class BaseSampler(ABC):

  @classmethod
  def from_worker_factory(cls, worker_factory: WorkerFactory, agents: Union[BasePolicy, List], envs: List[BaseEnv])\
  -> 'BaseSampler':
    """
    Create a sampler.

    Args:
      worker_factory (WorkerFactory): Pickleable factory for creating workers.
      agents (Union[BasePolicy, List]): Agents used to collect episodes.
      envs (List[BaseEnv]): A list of environments.

    Returns:
      BaseSampler
    """
    raise NotImplementedError

  @abstractmethod
  def obtain_samples(self, current_iteration: int, num_samples: int, agent_update, env_update = None) -> EpisodeBatch:
    """
    Collect at least a given number of transitions.

    Args:
      current_iteration (int): Current iteration number, using this argument is deprecated.
      num_samples (int): Minimum number of time steps to sample.
      agent_update (object): Value which will be passed into the agent_update_fn before sampling episodes.
      env_update (object): Value which will be passed into the env_update_fn before sampling episodes.

    Returns:
      EpisodeBatch
    """
    raise NotImplementedError

  @abstractmethod
  def shutdown_worker(self):
    """
    Terminate the workers if necessary.

    Returns:
      None
    """
    raise NotImplementedError
