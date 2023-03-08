from abc import ABC, abstractmethod
from typing import List, Union

import torch.nn as nn

from rl.structs import EpisodeBatch
from rl.envs.base_env import BaseEnv
from rl.networks.policies.base_policy import BasePolicy

from .workers import WorkerFactory


class BaseSampler(ABC):

  @classmethod
  def from_worker_factory(cls, worker_factory: WorkerFactory, agents: Union[BasePolicy, List], envs: List[BaseEnv]):
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
  def obtain_samples(self, num_samples: int, agent_policy: nn.Module, env_update = None):
    """
    Collect at least a given number transitions (timesteps).

    Args:
      num_samples (int): Minimum number of transitions / timesteps to sample.
      agent_policy (nn.Module): Value which will be passed into the `agent_update_fn` before sampling episodes.
        If a list is passed in, it must have length exactly `factory.n_workers`, and will be spread across the workers.
      env_update (object): Value which will be passed into the `env_update_fn` before sampling episodes. If a list is
        passed in, it must have length exactly `factory.n_workers`, and will be spread across the workers.

    Returns:
      EpisodeBatch
    """
    raise NotImplementedError

  @abstractmethod
  def shutdown_workers(self):
    """
    Terminate the workers if necessary.

    Returns:
      None
    """
    raise NotImplementedError
