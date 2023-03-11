from typing import List
from torch.optim import Optimizer
from rl.utils.modules.batch_dataset import BatchDataset


class WrappedOptimizer:

  def __init__(self, optimizer: Optimizer, max_optimization_epochs: int = 1, minibatch_size: int = None):
    """
    A wrapper class to handle torch.optim.Optimizer.

    Args:
      optimizer (Optimizer): PyTorch optimizer.
      max_optimization_epochs (int): Maximum number of epochs for update.
      minibatch_size (int): Batch size for optimization.
    """
    self._optimizer = optimizer
    self._max_optimization_epochs = max_optimization_epochs
    self._minibatch_size = minibatch_size
    pass

  def get_minibatch(self, *inputs):
    """
    Yields a batch of inputs.

    Args:
      *inputs (List[torch.Tensor]): A list of inputs.

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

  def step(self, closure: callable = None):
    """
    Performs a single optimization step.

    Arguments:
      **closure (callable, optional): A closure that reevaluates the model and returns the loss.
    """
    if closure:
      self._optimizer.step(closure)
    else:
      self._optimizer.step()
    pass
