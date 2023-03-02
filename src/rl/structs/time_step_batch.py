from typing import List, Dict
import warnings
import numpy as np
from dataclasses import dataclass

from .step_type import StepType
from .env_spec import EnvSpec

from src.rl.utils.functions import space_soft_contains


@dataclass(frozen = True)
class TimeStepBatch:
  """
  A tuple representing a batch of TimeSteps.

  Attributes:
    env_spec (EnvSpec): Environment specifications.
    episode_infos (dict[str, np.ndarray]): A dict of numpy arrays containing the episode-level information of each
      episode. Each value of this dict should be a numpy array of shape :math:`(N, S^*)`.
      For example, in goal-conditioned reinforcement learning this could contain the goal state for each episode.
    observations (numpy.ndarray): Non-flattened array of observations. Typically has shape (batch_size, S^*) (the
      unflattened state space of the current environment).
    actions (numpy.ndarray): Non-flattened array of actions. Must have shape (batch_size, S^*) (the unflattened action
      space of the current environment).
    rewards (numpy.ndarray): Array of rewards of shape (batch_size, 1).
    env_infos (dict): A dict arbitrary environment state information.
    agent_infos (dict): A dict of arbitrary agent state information. For example, this may contain the hidden states
      from an RNN policy.
    step_types (numpy.ndarray): A numpy array of `StepType with shape (batch_size,) containing the time step types for
      all transitions in this batch.

  Raises:
      ValueError: If any of the above attributes do not conform to their prescribed types and shapes.
  """
  env_spec: EnvSpec
  episode_infos: Dict[str, np.ndarray or dict]
  observations: np.ndarray
  actions: np.ndarray
  rewards: np.ndarray
  next_observations: np.ndarray
  agent_infos: Dict[str, np.ndarray or dict]
  env_infos: Dict[str, np.ndarray or dict]
  step_types: np.ndarray

  def __post_init__(self):
    """
    Runs integrity checking after __init__.
    """
    self.check_timestep_batch(self, np.ndarray)

  @classmethod
  def concatenate(cls, *batches):
    """
    Concatenate two or more :class:`TimeStepBatch`s.

    Args:
      batches (list[TimeStepBatch]): Batches to concatenate.

    Returns:
      TimeStepBatch: The concatenation of the batches.

    Raises:
      ValueError: If no TimeStepBatches are provided.
    """
    if len(batches) < 1:
      raise ValueError('Please provide at least one TimeStepBatch to concatenate')

    episode_infos = {
      k: np.concatenate([b.episode_infos[k] for b in batches])
      for k in batches[0].episode_infos.keys()
    }

    env_infos = {
      k: np.concatenate([b.env_infos[k] for b in batches])
      for k in batches[0].env_infos.keys()
    }

    agent_infos = {
      k: np.concatenate([b.agent_infos[k] for b in batches])
      for k in batches[0].agent_infos.keys()
    }

    return cls(
      env_spec = batches[0].env_spec,
      episode_infos = episode_infos,
      observations = np.concatenate([batch.observations for batch in batches]),
      actions = np.concatenate([batch.actions for batch in batches]),
      rewards = np.concatenate([batch.rewards for batch in batches]),
      next_observations = np.concatenate([batch.next_observations for batch in batches]),
      env_infos = env_infos,
      agent_infos = agent_infos,
      step_types = np.concatenate([batch.step_types for batch in batches])
    )

  def split(self) -> List['TimeStepBatch']:
    """
    Split a :class:`~TimeStepBatch` into a list of :class:`~TimeStepBatch`s.

    The opposite of concatenate.

    Returns:
      list[TimeStepBatch]: A list of :class:`TimeStepBatch`s, with one :class:`~TimeStep` per :class:`~TimeStepBatch`.
    """
    time_steps = []

    for i in range(len(self.rewards)):
      time_step = TimeStepBatch(
        episode_infos = {k: np.asarray([v[i]]) for (k, v) in self.episode_infos.items()},
        env_spec = self.env_spec,
        observations = np.asarray([self.observations[i]]),
        actions = np.asarray([self.actions[i]]),
        rewards = np.asarray([self.rewards[i]]),
        next_observations = np.asarray([self.next_observations[i]]),
        env_infos = {k: np.asarray([v[i]])for (k, v) in self.env_infos.items()},
        agent_infos = {k: np.asarray([v[i]]) for (k, v) in self.agent_infos.items()},
        step_types = np.asarray([self.step_types[i]], dtype = StepType)
      )

      time_steps.append(time_step)

    return time_steps

  def to_time_step_list(self) -> List[Dict[str, np.ndarray]]:
    """
    Convert the batch into a list of dictionaries.
    Breaks the :class:`~TimeStepBatch` into a list of single time step
    sample dictionaries. len(rewards) (or the number of discrete time step)
    dictionaries are returned

    Returns:
      list[dict[str, np.ndarray or dict[str, np.ndarray]]]
    """
    samples = []
    for i in range(len(self.rewards)):
      samples.append({
        'episode_infos': {k: np.asarray([v[i]]) for (k, v) in self.episode_infos.items()},
        'observations': np.asarray([self.observations[i]]),
        'actions': np.asarray([self.actions[i]]),
        'rewards': np.asarray([self.rewards[i]]),
        'next_observations': np.asarray([self.next_observations[i]]),
        'env_infos': {k: np.asarray([v[i]]) for (k, v) in self.env_infos.items()},
        'agent_infos': {k: np.asarray([v[i]]) for (k, v) in self.agent_infos.items()},
        'step_types': np.asarray([self.step_types[i]])
      })

    return samples

  @property
  def terminals(self):
    """
    Get an array of boolean indicating ternianal information.

    Returns:
      numpy.ndarray
    """
    return np.array([s == StepType.TERMINAL for s in self.step_types])

  @classmethod
  def from_time_step_list(cls, ts_samples):
    """
    Create a :class:`~TimeStepBatch` from a list of time step dictionaries.

    Returns:
      TimeStepBatch: The concatenation of samples.

    Raises:
      ValueError: If no dicts are provided.
    """
    if len(ts_samples) < 1:
      raise ValueError('Please provide at least one dict')

    ts_batches = [
      TimeStepBatch(
        env_spec = sample['env_spec'],
        episode_infos = sample['episode_infos'],
        observations = sample['observations'],
        actions = sample['actions'],
        rewards = sample['rewards'],
        next_observations = sample['next_observations'],
        env_infos = sample['env_infos'],
        agent_infos = sample['agent_infos'],
        step_types = sample['step_types']
      ) for sample in ts_samples
    ]

    return TimeStepBatch.concatenate(*ts_batches)

  @staticmethod
  def check_timestep_batch(batch, array_type: type = np.ndarray, ignored_fields: set = ()):
    """
    Check a TimeStepBatch of any array type that has .shape.

    Args:
      batch (TimeStepBatch): Batch of timesteps.
      array_type (type): Array type.
      ignored_fields (set[str]): Set of fields to ignore checking on.

    Raises:
      ValueError: If an invariant of TimeStepBatch is broken.
    """
    fields = {
      field: getattr(batch, field)
      for field in [
        'env_spec', 'rewards', 'rewards', 'observations', 'actions',
        'next_observations', 'step_types', 'agent_infos', 'episode_infos',
        'env_infos'
      ] if field not in ignored_fields
    }

    env_spec = fields.get('env_spec', None)
    inferred_batch_size = None
    inferred_batch_size_field = None

    for field, value in fields.items():
      if field in ['observations', 'actions', 'rewards', 'next_observations', 'step_types'] and not \
      isinstance(value, array_type):
          raise ValueError(f'{field} is not of type {array_type!r}')

      if hasattr(value, 'shape'):
        if inferred_batch_size is None:
          inferred_batch_size = value.shape[0]
          inferred_batch_size_field = field
        elif value.shape[0] != inferred_batch_size:
          raise ValueError(f'{field} has batch size {value.shape[0]}, but must have batch size {inferred_batch_size} '
                           f'to match {inferred_batch_size_field}')

        if env_spec and field in ['observations', 'next_observations']:
          if not space_soft_contains(env_spec.observation_space, value[0]):
            raise ValueError(f'Each {field[:-1]} has shape {value[0].shape} but must match the observation_space '
                             f'{env_spec.observation_space}')

          if isinstance(value[0], np.ndarray) and not env_spec.observation_space.contains(value[0]):
            warnings.warn(f'Observation {value[0]!r} is outside observation_space {env_spec.observation_space}')

        if env_spec and field == 'actions':
          if not space_soft_contains(env_spec.action_space, value[0]):
            raise ValueError(f'Each {field[:-1]} has shape {value[0].shape} but must match the action_space '
                             f'{env_spec.action_space}')

        if field in ['rewards', 'step_types']:
          if value.shape != (inferred_batch_size,):
            raise ValueError(f'{field} has shape {value.shape} but must have batch size {inferred_batch_size} to match '
                             f'{inferred_batch_size_field}')

      if field in ['agent_infos', 'env_infos', 'episode_infos']:
        for key, val in value.items():
          if not isinstance(val, (array_type, dict)):
            raise ValueError(f'Entry {key!r} in {field} is of type {type(val)} but must be {array_type!r} or dict')

          if hasattr(val, 'shape') and val.shape[0] != inferred_batch_size:
            raise ValueError(f'Entry {key!r} in {field} has batch size {val.shape[0]} but must have batch size '
              f'{inferred_batch_size} to match {inferred_batch_size_field}')

      if field == 'step_types' and isinstance(value, np.ndarray) and value.dtype != StepType:
        raise ValueError(f'step_types has dtype {value.dtype} but must have dtype StepType')
