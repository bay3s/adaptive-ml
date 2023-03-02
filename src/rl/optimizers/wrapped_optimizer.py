from typing import Union, Tuple

import torch.nn as nn
import torch

from src.rl.utils.batch_dataset import BatchDataset
from src.rl.utils.functions import make_optimizer


class WrappedOptimizer:

  def __init__(self, optimizer: Union[torch.optim.Optimizer, Tuple], module: nn.Module,
               max_optimization_epochs: int = 1, minibatch_size: int = None):
    """
    A wrapper class to handle torch.optim.Optimizer.

    Args:
      optimizer (Union[type, tuple[type, dict]]): Type of optimizer  for policy. This can be an optimizer type such as
        `torch.optim.Adam` or a tuple of type and dictionary, where dictionary contains arguments to initialize the
        optimizer.
      module (torch.nn.Module): Module to be optimized.
      max_optimization_epochs (int): Maximum number of epochs for update.
      minibatch_size (int): Batch size for optimization.
    """
    self._optimizer = make_optimizer(optimizer, module = module)
    self._max_optimization_epochs = max_optimization_epochs
    self._minibatch_size = minibatch_size
    pass

  def get_minibatch(self, *inputs):
    """
    Yields a batch of inputs.

    Notes: P is the size of minibatch (self._minibatch_size)

    Args:
      *inputs (list[torch.Tensor]): A list of inputs. Each input has shape :math:`(N \dot [T], *)`.

    Yields:
      list[torch.Tensor]: A list batch of inputs. Each batch has shape :math:`(P, *)`.
    """
    batch_dataset = BatchDataset(inputs, self._minibatch_size)

    for _ in range(self._max_optimization_epochs):
      for dataset in batch_dataset.iterate():
        yield dataset

  def zero_grad(self) -> None:
    """
    Clears the gradients of all optimized :class:`torch.Tensor` s.
    """
    self._optimizer.zero_grad()
    pass

  def step(self, **closure):
    """
    Performs a single optimization step.

    Arguments:
      **closure (callable, optional): A closure that reevaluates the model and returns the loss.
    """
    self._optimizer.step(**closure)
    pass
