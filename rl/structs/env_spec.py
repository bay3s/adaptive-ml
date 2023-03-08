from dataclasses import dataclass

import akro


@dataclass(frozen = True)
class EnvSpec:
  """
  Describes the observations, actions, and time horizon of an MDP.

  Args:
    observation_space (akro.Space): The observation space of the env.
    action_space (akro.Space): The action space of the env.
    max_episode_length (int): The maximum number of steps allowed in an episode.
  """
  observation_space: akro.Space
  action_space: akro.Space
  max_episode_length: int
