from abc import abstractmethod, ABC

import torch.nn as nn


class BaseValueFunction(ABC, nn.Module):

  def __init__(self, env_spec, unique_id):
    """
    Base class for all baselines.

    Args:
      env_spec (EnvSpec): Environment specification.
      unique_id (str): Value function name, also the variable scope.
    """
    super(BaseValueFunction, self).__init__()

    self._mdp_spec = env_spec
    self.unique_id = unique_id
    pass

  @abstractmethod
  def compute_loss(self, obs, returns):
    """
    Compute mean value of loss.

    Args:
      obs (torch.Tensor): Observation from the environment
      returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.

    Returns:
      torch.Tensor: Calculated negative mean scalar value of objective (float).
    """
    raise NotImplementedError
