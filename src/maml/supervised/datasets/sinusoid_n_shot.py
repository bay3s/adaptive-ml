import numpy as np
import torch

from src.maml.supervised.datasets import Sinusoid


class SinusoidNShot:

  DATASET_FILE_NPY = 'omniglot_n_shot.npy'

  def __init__(self, batch_size: int, K_shot: int, K_query: int, device: str):
    """
    Initialize the Sinusoid N-shot dataset.

    Args:
      batch_size (int): Batch size for each of the
      K_shot (int): K-shot for the training step in meta-training.
      K_query (int): K-query for the testing step in meta-training.
      K_query (int): K-query for the testing step in meta-training.
      device (str): Device to be used for PyTorch tensors.
    """
    self.batch_size = batch_size
    self.K_shot = K_shot
    self.K_query = K_query
    pass

  @staticmethod
  @torch.no_grad()
  def generate_sinusoid() -> Sinusoid:
    """
    Samples amplitude and phase, then returns a Sine Wave function corresponding to it.

    Returns:
        Returns a neural network that is tuned for few-shot learning.
    """
    amp = torch.rand(1).item() * 4.9 + 0.1
    phase = torch.rand(1).item() * np.pi

    return Sinusoid(amp, phase)

  def next(self) -> [np.array, np.array, np.array, np.array]:
    """
    Retrieve the next batch in the dataset.

    Returns:
      [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    """
    sinsusoid_functions = [self.generate_sinusoid() for _ in range(self.batch_size)]
    x_support_batch, y_support_batch, x_query_batch, y_query_batch = list(), list(), list(), list()

    for i, task in enumerate(sinsusoid_functions):
      x_support, y_support = task.sample(self.K_shot)
      x_support_batch.append(x_support)
      y_support_batch.append(y_support)

      x_query, y_query = task.sample(self.K_query)
      x_query_batch.append(x_query)
      y_query_batch.append(y_query)
      continue

    x_support_batch = np.array(x_support_batch).astype(np.float32)
    y_support_batch = np.array(y_support_batch).astype(np.float32)

    x_query_batch = np.array(x_query_batch).astype(np.float32)
    y_query_batch = np.array(y_query_batch).astype(np.float32)

    return x_support_batch, y_support_batch, x_query_batch, y_query_batch
