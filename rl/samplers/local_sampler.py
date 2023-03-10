import copy
import torch.nn as nn
from typing import Union, List

from rl.networks.policies.base_policy import BasePolicy
from rl.samplers.base_sampler import BaseSampler
from rl.samplers.workers import WorkerFactory
from rl.structs import EpisodeBatch
from rl.envs.base_env import BaseEnv


class LocalSampler(BaseSampler):

  def __init__(self, agents: Union[List, BasePolicy], envs: Union[List[BaseEnv], BaseEnv], worker_factory: WorkerFactory):
    """
    Local Sampler runs workers in the main process.

    Args:
      agents (BasePolicy or List[BasePolicy]): Agent(s) to use to sample episodes.
      envs (Environment or List[Environment]): Environment from which episodes are sampled.
      worker_factory (WorkerFactory): Factory for creating workers.
    """
    self._factory = worker_factory
    self._agents = self._factory.prepare_worker_messages(agents)
    self._envs = self._factory.prepare_worker_messages(envs, preprocess = copy.deepcopy)

    self._workers = [self._factory(i) for i in range(self._factory.n_workers)]

    for worker, agent, env in zip(self._workers, self._agents, self._envs):
      worker.update_agent(agent)
      worker.update_env(env)

    self.total_env_steps = 0
    pass

  @classmethod
  def from_worker_factory(cls, worker_factory: WorkerFactory, agents: list, envs: list):
    """
    Construct this sampler.

    Args:
      worker_factory (WorkerFactory): Pickleable factory for creating workers.
      agents (Agent or List[Agent]): Agent(s) to use to sample episodes.
      envs (Environment or List[Environment]): Environment from which episodes are sampled.

    Returns:
        BaseSampler
    """
    return cls(agents, envs, worker_factory = worker_factory)

  def _update_workers(self, agent_update, env_update):
    """
    Apply updates to the workers.

    Args:
      agent_update (object): Value which will be passed into the `agent_update_fn` before sampling episodes.
      env_update (object): Value which will be passed into the `env_update_fn` before sampling episodes.
        If a list is passed in, it must have length exactly `factory.n_workers`, and will be spread across the workers.
    """
    agent_updates = self._factory.prepare_worker_messages(agent_update)
    env_updates = self._factory.prepare_worker_messages(env_update, preprocess = copy.deepcopy)

    for worker, agent_up, env_up in zip(self._workers, agent_updates, env_updates):
      worker.update_agent(agent_up)
      worker.update_env(env_up)

  def obtain_samples(self, num_samples: int, agent_policy: BasePolicy, env_update = None):
    """
    Collect at least a given number transitions (timesteps).

    Args:
      num_samples (int): Minimum number of transitions / timesteps to sample.
      agent_policy (nn.Module): Value which will be passed into the `agent_update_fn` before sampling episodes.
      env_update (object): Value which will be passed into the `env_update_fn` before sampling episodes.

    Returns:
      EpisodeBatch
    """
    self._update_workers(agent_policy, env_update)
    batches = []
    completed_samples = 0

    while True:
      for worker in self._workers:
        batch = worker.rollout()
        completed_samples += len(batch.actions)
        batches.append(batch)

        if completed_samples >= num_samples:
          samples = EpisodeBatch.concatenate(*batches)
          self.total_env_steps += sum(samples.lengths)

          return samples

  def obtain_exact_episodes(self, n_eps_per_worker, agent_update, env_update = None):
    """
    Sample an exact number of episodes per worker.

    Args:
      n_eps_per_worker (int): Exact number of episodes to gather for each worker.
      agent_update (object): Value which will be passed into the `agent_update_fn` before sampling episodes.
      env_update (object): Value which will be passed into the `env_update_fn` before samplin episodes.

    Returns:
        EpisodeBatch
    """
    self._update_workers(agent_update, env_update)
    batches = []

    for worker in self._workers:
      for _ in range(n_eps_per_worker):
        batch = worker.rollout()
        batches.append(batch)

    samples = EpisodeBatch.concatenate(*batches)
    self.total_env_steps += sum(samples.lengths)

    return samples

  def shutdown_workers(self) -> None:
    """
    Shutdown the workers.
    """
    for worker in self._workers:
      worker.shutdown()

  def __getstate__(self):
    """
    Get the pickle state.

    Returns:
      dict: The pickled state.
    """
    state = self.__dict__.copy()
    state['_workers'] = None

    return state

  def __setstate__(self, state):
    """
    Unpickle the state.

    Args:
      state (dict): Unpickled state.
    """
    self.__dict__.update(state)
    self._workers = [
      self._factory(i) for i in range(self._factory.n_workers)
    ]

    for worker, agent, env in zip(self._workers, self._agents, self._envs):
      worker.update_agent(agent)
      worker.update_env(env)