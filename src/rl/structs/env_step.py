from typing import Dict
import numpy as np
from dataclasses import dataclass
from .step_type import StepType


@dataclass
class EnvStep:
  """
  A tuple representing a single step returned by the environment.

  Attributes:
    env_name (str): Name of the environment in which this step was taken.
    action (numpy.ndarray): A numpy array of shape :math:`(A^*)` containing the action for this time step.
      These must conform to :obj:`EnvStep.action_space`. `None` if `step_type` is `StepType.FIRST`, i.e. at
      the start of a sequence.
    reward (float): A float representing the reward for taking the action given the observation, at this time
      step. `None` if `step_type` is `StepType.FIRST`, i.e. at the start of a sequence.
    observation (numpy.ndarray): A numpy array of shape :math:`(O^*)` containing the observation for this time step
      in the environment. These must conform to :obj:`EnvStep.observation_space`. The observation after applying the
      action.
    env_info (dict): A dict containing environment state information.
    step_type (StepType): a `StepType` enum value. Can either be StepType.FIRST, StepType.MID, StepType.TERMINAL,
      StepType.TIMEOUT.
  """

  env_name: str
  action: np.ndarray
  reward: np.ndarray
  observation: np.ndarray
  env_info: Dict[str, np.ndarray or dict]
  step_type: StepType

  @property
  def first(self):
    """
    bool: Whether this `TimeStep` is the first of a sequence.
    """
    return self.step_type is StepType.FIRST

  @property
  def mid(self):
    """
    bool: Whether this `TimeStep` is in the mid of a sequence.
    """
    return self.step_type is StepType.MID

  @property
  def terminal(self):
    """
    bool: Whether this `TimeStep` records a termination condition.
    """
    return self.step_type is StepType.TERMINAL

  @property
  def timeout(self):
    """
    bool: Whether this `TimeStep` records a timeout condition.
    """
    return self.step_type is StepType.TIMEOUT

  @property
  def last(self):
    """
    bool: Whether this `TimeStep` is the last of a sequence.
    """
    return self.step_type is StepType.TERMINAL or self.step_type is StepType.TIMEOUT
