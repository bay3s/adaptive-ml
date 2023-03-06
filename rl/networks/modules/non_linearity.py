from typing import Callable
import copy

import torch
import torch.nn as nn


class NonLinearity(nn.Module):

  def __init__(self, non_linear: Callable):
    """
    Wrapper class for non-linear function or module.

    Args:
      non_linear (callable or type): Non-linear function or type to be wrapped.
    """
    super().__init__()

    if isinstance(non_linear, type):
      self.module = non_linear()
    elif callable(non_linear):
      self.module = copy.deepcopy(non_linear)
    else:
      raise ValueError('Non linear function {} is not supported'.format(non_linear))

  def forward(self, input_value: torch.Tensor) -> torch.Tensor:
    """
    Forward method.

    Args:
      input_value (torch.Tensor): Input values

    Returns:
      torch.Tensor: Output value
    """
    return self.module(input_value)

  def __repr__(self):
    """
    Object representation method.
    """
    return repr(self.module)
