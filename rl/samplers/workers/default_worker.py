import numpy as np
from collections import defaultdict

from rl.structs import EpisodeBatch, StepType
from rl.utils.functions.device_functions import set_seed
from rl.networks.policies.base_policy import BasePolicy
from rl.envs.base_env import BaseEnv

from .base_worker import BaseWorker
from rl.samplers.update_handlers.base_env_update_handler import BaseEnvUpdateHandler


class DefaultWorker(BaseWorker):

  def __init__(self, *, seed: int, max_episode_length: int, worker_number: int):
    """
    Initialize a worker.

    Args:
      seed (int): The seed to use to intialize random number generators.
      max_episode_length (int or float): The maximum length of episodes which will be sampled.
      worker_number (int): The number of the worker this update is occurring in.
    """
    super().__init__(
      seed = seed,
      max_episode_length = max_episode_length,
      worker_number = worker_number,
    )

    self.agent = None
    self.env = None
    self._env_steps = []
    self._observations = []
    self._last_observations = []
    self._agent_infos = defaultdict(list)

    self._lengths = []
    self._prev_obs = None
    self._eps_length = 0
    self._episode_infos = defaultdict(list)

    self.worker_init()
    pass

  def worker_init(self) -> None:
    """
    Initialize a worker.

    Returns:
      None
    """
    if self._seed is not None:
      set_seed(self._seed + self._worker_number)

  def update_agent(self, agent_policy: BasePolicy) -> None:
    """
    Update the worker's agent, using agent_policy.

    Args:
      agent_policy (BasePolicy): An agent update.
    """
    self.agent = agent_policy

  def update_env(self, env_update):
    """
    Update the worker's env, using env_update.

    Args:
      env_update (object): An environment update.
    """
    self.env, _ = self._apply_env_update(self.env, env_update)

  def rollout(self) -> EpisodeBatch:
    """
    Sample a single episode of the agent in the environment.

    Returns:
      EpisodeBatch
    """
    self.start_episode()
    while not self.step_episode():
      pass

    return self.collect_episode()

  def start_episode(self):
    """
    Begin a new episode.
    """
    self._eps_length = 0
    self._prev_obs, episode_info = self.env.reset()

    for k, v, in episode_info.items():
      self._episode_infos[k].append(v)

    self.agent.reset()
    pass

  def step_episode(self) -> bool:
    """
    Take a single time-step in the current episode.

    Returns true if the episode is done.

    Returns:
      bool
    """
    if self._eps_length < self._max_episode_length:
      a, agent_info = self.agent.get_action(self._prev_obs)

      es = self.env.step(a)
      self._observations.append(self._prev_obs)
      self._env_steps.append(es)

      for k, v in agent_info.items():
        self._agent_infos[k].append(v)

      self._eps_length += 1

      if not es.terminal:
        self._prev_obs = es.observation
        return False

    self._lengths.append(self._eps_length)
    self._last_observations.append(self._prev_obs)

    return True

  def collect_episode(self) -> EpisodeBatch:
    """
    Collect the current episode, clearing the internal buffer.

    Returns:
      EpisodeBatch
    """
    observations = self._observations
    self._observations = []
    last_observations = self._last_observations
    self._last_observations = []

    actions = []
    rewards = []
    env_infos = defaultdict(list)
    step_types = []

    for es in self._env_steps:
      actions.append(es.action)
      rewards.append(es.reward)
      step_types.append(es.step_type)

      for k, v in es.env_info.items():
        env_infos[k].append(v)

    self._env_steps = []
    agent_infos = self._agent_infos
    self._agent_infos = defaultdict(list)

    for k, v in agent_infos.items():
      agent_infos[k] = np.asarray(v)

    for k, v in env_infos.items():
      env_infos[k] = np.asarray(v)

    episode_infos = self._episode_infos
    self._episode_infos = defaultdict(list)
    for k, v in episode_infos.items():
      episode_infos[k] = np.asarray(v)

    lengths = self._lengths
    self._lengths = []

    return EpisodeBatch(
      env_spec = self.env.spec,
      episode_infos = episode_infos,
      observations = np.asarray(observations),
      last_observations = np.asarray(last_observations),
      actions = np.asarray(actions),
      rewards = np.asarray(rewards),
      step_types = np.asarray(step_types, dtype = StepType),
      env_infos = dict(env_infos),
      agent_infos = dict(agent_infos),
      lengths = np.asarray(lengths, dtype = 'i')
    )

  def shutdown(self):
    """
    Shutdown the worker.
    """
    self.env.close()
    pass

  def __getstate__(self):
    """
    Refuse to be pickled.

    Raises:
      ValueError: Always raised, since pickling Workers is not supported.
    """
    raise ValueError('Workers are not pickleable, please pickle the WorkerFactory instead.')

  @staticmethod
  def _apply_env_update(old_env, env_update):
    """
    This allows changing environments by passing the new environment as `env_update` into `obtain_samples`.

    Args:
      old_env (Environment): Environment to updated.
      env_update (Environment or EnvUpdate or None): The environment to replace the existing env with.

    Returns:
      Environment: The updated environment.
      bool: True if there was an update made.

    Raises:
      TypeError: If env_update is not one of the documented types.
    """
    if env_update is not None:
      if isinstance(env_update, BaseEnvUpdateHandler):
        return env_update(old_env), True
      elif isinstance(env_update, BaseEnv):
        if old_env is not None:
          old_env.close()
        return env_update, True
      else:
        raise TypeError('Unknown environment update type.')
    else:
      return old_env, False
