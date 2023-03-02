import numpy as np
from typing import Dict
from dataclasses import dataclass
from .step_type import StepType


@dataclass(frozen = True)
class TimeStep:
  """
  A single TimeStep in an environment.

  A :class:`~TimeStep` represents a single sample when an agent interacts with an environment. It describes as
  SARS (State–action–reward–state) tuple that characterizes the evolution of an MDP.

  Attributes:
    env_name (str): Name of the environment from which data was sampled.
    episode_info (dict[str, np.ndarray]): A dict of numpy arrays of shape
      :math:`(S*^,)` containing episode-level information of each
      episode.  For example, in goal-conditioned reinforcement learning
      this could contain the goal state for each episode.
    observation (numpy.ndarray): A numpy array of shape :math:`(O^*)`
      containing the observation for this time step in the
      environment. These must conform to
      :obj:`EnvStep.observation_space`.
      The observation before applying the action.
      `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a
      sequence.
    action (numpy.ndarray): A numpy array of shape :math:`(A^*)`
      containing the action for this time step. These must conform
      to :obj:`EnvStep.action_space`.
      `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a
      sequence.
    reward (float): A float representing the reward for taking the action given the observation, at this time step.
      `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a sequence.
    next_observation (numpy.ndarray): A numpy array of shape :math:`(O^*)` containing the observation for this time
      step in the environment. These must conform to :obj:`EnvStep.observation_space`.
    env_info (dict): A dict arbitrary environment state information.
    agent_info (dict): A dict of arbitrary agent
        state information. For example, this may contain the hidden states
        from an RNN policy.
    step_type (StepType): a :class:`~StepType` enum value. Can be one of
        :attribute:`~StepType.FIRST`, :attribute:`~StepType.MID`,
        :attribute:`~StepType.TERMINAL`, or :attribute:`~StepType.TIMEOUT`.
  """

  env_name: str
  episode_info: Dict[str, np.ndarray]
  observation: np.ndarray
  action: np.ndarray
  reward: float
  next_observation: np.ndarray
  env_info: Dict[str, np.ndarray]
  agent_info: Dict[str, np.ndarray]
  step_type: StepType

  @property
  def first(self):
    """
    bool: Whether this step is the first of its episode.
    """
    return self.step_type is StepType.FIRST

  @property
  def mid(self):
    """
    bool: Whether this step is in the middle of its episode.
    """
    return self.step_type is StepType.MID

  @property
  def terminal(self):
    """
    bool: Whether this step records a termination condition.
    """
    return self.step_type is StepType.TERMINAL

  @property
  def timeout(self):
    """
    bool: Whether this step records a timeout condition.
    """
    return self.step_type is StepType.TIMEOUT

  @property
  def last(self):
    """
    bool: Whether this step is the last of its episode.
    """
    return self.step_type is StepType.TERMINAL or self.step_type is StepType.TIMEOUT

  @classmethod
  def from_env_step(cls, env_step, last_observation, agent_info, episode_info):
    """
    Create a TimeStep from an EnvStep.

    Args:
      env_step (EnvStep): the env step returned by the environment.
      last_observation (numpy.ndarray): A numpy array of shape  :math:`(O^*)` containing the observation for this time
        step in the environment. These must conform to :attr:`EnvStep.observation_space`. The observation before
        applying the action.
      agent_info (dict): A dict of arbitrary agent state information.
      episode_info (dict): A dict of arbitrary information associated with the whole episode.

    Returns:
      TimeStep: The TimeStep with all information of EnvStep plus the agent info.
    """
    return cls(
      env_name = env_step.env_name,
      episode_info = episode_info,
      observation = last_observation,
      action = env_step.action,
      reward = env_step.reward,
      next_observation = env_step.observation,
      env_info = env_step.env_info,
      agent_info = agent_info,
      step_type = env_step.step_type
    )
