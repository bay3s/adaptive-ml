from typing import List
import numpy as np
import torch


class BatchDataset:

  def __init__(self, inputs: List[torch.Tensor], batch_size: int, extra_inputs: List = None):
    """
    Batch dataset required for the optimizers.

    Args:
      inputs (List[torch.Tensor]): A list of inputs.
      batch_size (int): Batch size for batching.
      extra_inputs (List): Additional inputs.
    """
    self._inputs = [i for i in inputs]

    if extra_inputs is None:
      extra_inputs = []

    self._extra_inputs = extra_inputs
    self._batch_size = batch_size

    if batch_size is not None:
      self._ids = np.arange(self._inputs[0].shape[0])
      self.update()

  @property
  def number_batches(self) -> int:
    """
    Returns the number of batches based on the shape of the input and the batch size provided.

    Returns:
      int
    """
    if self._batch_size is None:
      return 1

    return int(np.ceil(self._inputs[0].shape[0] * 1.0 / self._batch_size))

  def iterate(self, update: bool = True):
    """
    Iterate over the provided inputs.

    Args:
      update (bool): Whether to update the ordering of the dataset once a batch is yielded.

    Returns:
      List
    """
    if self._batch_size is None:
      yield list(self._inputs) + list(self._extra_inputs)
    else:
      for itr in range(self.number_batches):
        batch_start = itr * self._batch_size
        batch_end = (itr + 1) * self._batch_size
        batch_ids = self._ids[batch_start:batch_end]
        batch = [d[batch_ids] for d in self._inputs]
        yield list(batch) + list(self._extra_inputs)

      if update:
        self.update()

  def update(self) -> None:
    """
    Shuffle ids for each of the inputs.

    Returns:
      None
    """
    np.random.shuffle(self._ids)
