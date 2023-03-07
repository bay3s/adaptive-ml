import numpy as np
from dataclasses import dataclass

from .time_step_batch import TimeStepBatch
from .step_type import StepType


from rl.utils.functions.rl_functions import space_soft_contains
from rl.utils.functions.preprocessing_functions import (
  pad_batch_array,
  stack_tensor_dict_list,
  concat_tensor_dict_list,
  slice_nested_dict
)


@dataclass(frozen = True)
class EpisodeBatch(TimeStepBatch):
  """
  A tuple representing a batch of whole episodes.

  A :class:`~EpisodeBatch` represents a batch of whole episodes, produced
  when one or more agents interacts with one or more environments.
  +-----------------------+-------------------------------------------------+
  | Symbol                | Description                                     |
  +=======================+=================================================+
  | :math:`N`             | Episode batch dimension                         |
  +-----------------------+-------------------------------------------------+
  | :math:`[T]`           | Variable-length time dimension of each          |
  |                       | episode                                         |
  +-----------------------+-------------------------------------------------+
  | :math:`S^*`           | Single-step shape of a time-series tensor       |
  +-----------------------+-------------------------------------------------+
  | :math:`N \bullet [T]` | A dimension computed by flattening a            |
  |                       | variable-length time dimension :math:`[T]` into |
  |                       | a single batch dimension with length            |
  |                       | :math:`sum_{i \in N} [T]_i`                     |
  +-----------------------+-------------------------------------------------+
  Attributes:
    env_spec (EnvSpec): Specification for the environment from which this data was sampled.
    episode_infos (dict[str, np.ndarray]): A dict of numpy arrays containing the episode-level information of each
      episode. Each value of this dict should be a numpy array of shape :math:`(N, S^*)`. For example, in
      goal-conditioned reinforcement learning this could contain the goal state for each episode.
    observations (numpy.ndarray): A numpy array of shape :math:`(N \bullet [T], O^*)` containing the (possibly
      multi-dimensional) observations for all time steps in this batch. These must conform to
      :obj:`EnvStep.observation_space`.
    last_observations (numpy.ndarray): A numpy array of shape :math:`(N, O^*)` containing the last observation
      of each episode. This is necessary since there are one more observations than actions every episode.
    actions (numpy.ndarray): A  numpy array of shape :math:`(N \bullet [T], A^*)` containing the (possibly
      multi-dimensional) actions for all time steps in this batch. These must conform to :obj:`EnvStep.action_space`.
    rewards (numpy.ndarray): A numpy array of shape :math:`(N \bullet [T])` containing the rewards for all time steps
      in this batch.
    env_infos (dict[str, np.ndarray]): A dict of numpy arrays arbitrary environment state information. Each value of
      this dict should be a numpy array of shape :math:`(N \bullet [T])` or :math:`(N \bullet [T], S^*)`.
    agent_infos (dict[str, np.ndarray]): A dict of numpy arrays arbitrary agent state information. Each value of this
      dict should be a numpy array of shape :math:`(N \bullet [T])` or :math:`(N \bullet [T], S^*)`.  For example,
      this may contain the hidden states from an RNN policy.
    step_types (numpy.ndarray): A numpy array of `StepType with shape :math:`(N \bullet [T])` containing the time step
      types for all transitions in this batch.
    lengths (numpy.ndarray): An integer numpy array of shape :math:`(N,)` containing the length of each episode in this
      batch. This may be used to reconstruct the individual episodes.
  Raises:
    ValueError: If any of the above attributes do not conform to their prescribed types and shapes.
  """
  episode_infos_by_episode: np.ndarray
  last_observations: np.ndarray
  lengths: np.ndarray

  def __init__(self, env_spec, episode_infos, observations, last_observations, actions, rewards, env_infos, agent_infos,
               step_types, lengths):  # noqa: D102
    # lengths
    if len(lengths.shape) != 1:
      raise ValueError(f'lengths has shape {lengths.shape} but must be a ternsor of shape (N,)')

    if not (lengths.dtype.kind == 'u' or lengths.dtype.kind == 'i'):
      raise ValueError(f'lengths has dtype {lengths.dtype}, but must have an integer dtype')

    n_episodes = len(lengths)

    # Check episode_infos and last_observations here instead of checking
    # episode_infos and next_observations in check_timestep_batch.

    for key, val in episode_infos.items():
      if not isinstance(val, np.ndarray):
        raise ValueError(f'Entry {key!r} in episode_infos is of type {type(val)!r} but must be of type {np.ndarray!r}')
      if hasattr(val, 'shape'):
        if val.shape[0] != n_episodes:
          raise ValueError(f'Entry {key!r} in episode_infos has batch size {val.shape[0]}, but must have batch size '
            f'{n_episodes} to match the number of episodes')

    if not isinstance(last_observations, np.ndarray):
      raise ValueError(f'last_observations is not of type {np.ndarray!r}')

    if last_observations.shape[0] != n_episodes:
      raise ValueError(f'last_observations has batch size {last_observations.shape[0]} but must have  batch size '
                       f'{n_episodes} to match the number of episodes')

    if not space_soft_contains(env_spec.observation_space, last_observations[0]):
      raise ValueError(f'last_observations must have the same number of entries as there are episodes  ({n_episodes}) '
                       f'but got data with shape {last_observations[0].shape} entries')

    object.__setattr__(self, 'last_observations', last_observations)
    object.__setattr__(self, 'lengths', lengths)
    object.__setattr__(self, 'env_spec', env_spec)
    object.__setattr__(self, 'episode_infos_by_episode', episode_infos)
    object.__setattr__(self, 'observations', observations)
    object.__setattr__(self, 'actions', actions)
    object.__setattr__(self, 'rewards', rewards)
    object.__setattr__(self, 'env_infos', env_infos)
    object.__setattr__(self, 'agent_infos', agent_infos)
    object.__setattr__(self, 'step_types', step_types)
    self.check_timestep_batch(self, np.ndarray, ignored_fields = {'next_observations', 'episode_infos'})

  @classmethod
  def concatenate(cls, *batches):
    """
    Create a EpisodeBatch by concatenating EpisodeBatches.

    Args:
      batches (list[EpisodeBatch]): Batches to concatenate.

    Returns:
      EpisodeBatch: The concatenation of the batches.
    """
    if __debug__:
      for b in batches:
        assert (set(b.env_infos.keys()) == set(
          batches[0].env_infos.keys()))
        assert (set(b.agent_infos.keys()) == set(
          batches[0].agent_infos.keys()))
    env_infos = {
      k: np.concatenate([b.env_infos[k] for b in batches])
      for k in batches[0].env_infos.keys()
    }
    agent_infos = {
      k: np.concatenate([b.agent_infos[k] for b in batches])
      for k in batches[0].agent_infos.keys()
    }
    episode_infos = {
      k: np.concatenate([b.episode_infos_by_episode[k] for b in batches])
      for k in batches[0].episode_infos_by_episode.keys()
    }

    return cls(
      episode_infos = episode_infos,
      env_spec = batches[0].env_spec,
      observations = np.concatenate(
        [batch.observations for batch in batches]),
      last_observations = np.concatenate(
        [batch.last_observations for batch in batches]),
      actions = np.concatenate([batch.actions for batch in batches]),
      rewards = np.concatenate([batch.rewards for batch in batches]),
      env_infos = env_infos,
      agent_infos = agent_infos,
      step_types = np.concatenate([batch.step_types for batch in batches]),
      lengths = np.concatenate([batch.lengths for batch in batches])
    )

  def _episode_ranges(self):
    """
    Iterate through start and stop indices for each episode.

    Yields:
      tuple[int, int]: Start index (inclusive) and stop index (exclusive).
    """
    start = 0
    for length in self.lengths:
      stop = start + length
      yield start, stop
      start = stop

  def split(self):
    """
    Split an EpisodeBatch into a list of EpisodeBatches. The opposite of concatenate.

    Returns:
      list[EpisodeBatch]: A list of EpisodeBatches, with one episode per batch.
    """
    episodes = []

    for i, (start, stop) in enumerate(self._episode_ranges()):
      eps = EpisodeBatch(
        env_spec = self.env_spec,
        episode_infos = slice_nested_dict(self.episode_infos_by_episode, i, i + 1),
        observations = self.observations[start:stop],
        last_observations = np.asarray([self.last_observations[i]]),
        actions = self.actions[start:stop],
        rewards = self.rewards[start:stop],
        env_infos = slice_nested_dict(self.env_infos, start, stop),
        agent_infos = slice_nested_dict(self.agent_infos, start, stop),
        step_types = self.step_types[start:stop],
        lengths = np.asarray([self.lengths[i]])
      )
      episodes.append(eps)

    return episodes

  def to_list(self):
    """
    Convert the batch into a list of dictionaries.

    Returns:
      list[dict[str, np.ndarray or dict[str, np.ndarray]]]
    """
    episodes = []
    for i, (start, stop) in enumerate(self._episode_ranges()):
      episodes.append({
        'episode_infos': {k: v[i:i + 1] for (k, v) in self.episode_infos.items()},
        'observations': self.observations[start:stop],
        'next_observations': np.concatenate((self.observations[1 + start:stop], [self.last_observations[i]])),
        'actions': self.actions[start:stop],
        'rewards': self.rewards[start:stop],
        'env_infos': {k: v[start:stop] for (k, v) in self.env_infos.items()},
        'agent_infos': {k: v[start:stop] for (k, v) in self.agent_infos.items()},
        'step_types': self.step_types[start:stop]
      })
    return episodes

  @classmethod
  def from_list(cls, env_spec, paths):
    """
    Create a EpisodeBatch from a list of episodes.

    Args:
      env_spec (EnvSpec): Specification for the environment from which this data was sampled.
      paths (list[dict[str, np.ndarray or dict[str, np.ndarray]]]): Paths sampled from the environment.
    """
    lengths = np.asarray([len(p['rewards']) for p in paths])
    if all(len(path['observations']) == length + 1 for (path, length) in zip(paths, lengths)):
      last_observations = np.asarray(
        [p['observations'][-1] for p in paths])
      observations = np.concatenate(
        [p['observations'][:-1] for p in paths])
    else:
      # The number of observations and timesteps must match.
      observations = np.concatenate([p['observations'] for p in paths])
      if paths[0].get('next_observations') is not None:
        last_observations = np.asarray(
          [p['next_observations'][-1] for p in paths])
      else:
        last_observations = np.asarray(
          [p['observations'][-1] for p in paths])

    stacked_paths = concat_tensor_dict_list(paths)
    episode_infos = stack_tensor_dict_list([path['episode_infos'] for path in paths])

    # Temporary solution. This logic is not needed if algorithms process
    # step_types instead of dones directly.
    if 'dones' in stacked_paths and 'step_types' not in stacked_paths:
      step_types = np.array([
        StepType.TERMINAL if done else StepType.MID
        for done in stacked_paths['dones']
      ], dtype = StepType)
      stacked_paths['step_types'] = step_types
      del stacked_paths['dones']

    return cls(
      env_spec = env_spec,
      episode_infos = episode_infos,
      observations = observations,
      last_observations = last_observations,
      actions = stacked_paths['actions'],
      rewards = stacked_paths['rewards'],
      env_infos = stacked_paths['env_infos'],
      agent_infos = stacked_paths['agent_infos'],
      step_types = stacked_paths['step_types'],
      lengths = lengths
    )

  @property
  def next_observations(self):
    """
    Get the observations seen after actions are performed. In an :class:`~EpisodeBatch`, next_observations don't need
    to be stored explicitly, since the next observation is already stored in the batch.

    Returns:
      np.ndarray
    """
    return np.concatenate(
      tuple([
        np.concatenate((eps.observations[1:], eps.last_observations))
        for eps in self.split()
      ]))

  @property
  def episode_infos(self):
    """
    Get the episode_infos. In an :class:`~EpisodeBatch`, episode_infos only need to be stored once per episode.
    However, the episode_infos field of :class:`~TimeStepBatch` has shape :math:`(N \bullet [T])`. This method
    expands episode_infos_by_episode (which have shape :math:`(N)`) to :math:`(N \bullet [T])`.

    Returns:
      dict[str, np.ndarray]
    """
    return {
      key: np.concatenate([
        np.repeat([v], length, axis = 0)
        for (v, length) in zip(val, self.lengths)
      ])
      for (key, val) in self.episode_infos_by_episode.items()
    }

  @property
  def padded_observations(self):
    """
    Padded observations.

    Returns:
      np.ndarray: Padded observations with shape of  :math:`(N, max_episode_length, O^*)`.
    """
    return pad_batch_array(self.observations, self.lengths, self.env_spec.max_episode_length)

  @property
  def padded_actions(self):
    """
    Padded actions.

    Returns:
      np.ndarray: Padded actions with shape of :math:`(N, max_episode_length, A^*)`.
    """
    return pad_batch_array(self.actions, self.lengths, self.env_spec.max_episode_length)

  @property
  def observations_list(self):
    """
    Split observations into a list.

    Returns:
      list[np.ndarray]: Splitted list.
    """
    obs_list = []
    for start, stop in self._episode_ranges():
      obs_list.append(self.observations[start:stop])
    return obs_list

  @property
  def actions_list(self):
    """
    Split actions into a list.

    Returns:
      list[np.ndarray]: Splitted list.
    """
    acts_list = []
    for start, stop in self._episode_ranges():
      acts_list.append(self.actions[start:stop])

    return acts_list

  @property
  def padded_rewards(self):
    """
    Padded rewards.

    Returns:
      np.ndarray: Padded rewards with shape of :math:`(N, max_episode_length)`.
    """
    return pad_batch_array(self.rewards, self.lengths, self.env_spec.max_episode_length)

  @property
  def valids(self):
    """
    An array indicating valid steps in a padded tensor.

    Returns:
      np.ndarray: the shape is :math:`(N, max_episode_length)`.
    """
    return pad_batch_array(np.ones_like(self.rewards), self.lengths,
                           self.env_spec.max_episode_length)

  @property
  def padded_next_observations(self):
    """
    Padded next_observations array.

    Returns:
      np.ndarray: Array of shape :math:`(N, max_episode_length, O^*)`
    """
    return pad_batch_array(self.next_observations, self.lengths,
                           self.env_spec.max_episode_length)

  @property
  def padded_step_types(self):
    """
    Padded step_type array.

    Returns:
      np.ndarray: Array of shape :math:`(N, max_episode_length)`
    """
    return pad_batch_array(self.step_types, self.lengths, self.env_spec.max_episode_length)

  @property
  def padded_agent_infos(self):
    """
    Padded agent infos.

    Returns:
      dict[str, np.ndarray]: Padded agent infos. Each value must have shape with :math:`(N, max_episode_length)` or
        :math:`(N, max_episode_length, S^*)`.
    """
    return {
      k: pad_batch_array(arr, self.lengths, self.env_spec.max_episode_length)
      for (k, arr) in self.agent_infos.items()
    }

  @property
  def padded_env_infos(self):
    """Padded env infos.
    Returns:
        dict[str, np.ndarray]: Padded env infos. Each value must have
            shape with :math:`(N, max_episode_length)` or
            :math:`(N, max_episode_length, S^*)`.
    """
    return {
      k: pad_batch_array(arr, self.lengths, self.env_spec.max_episode_length)
      for (k, arr) in self.env_infos.items()
    }
