import warnings
import random
import numpy as np
import torch


_USE_GPU = False
_DEVICE = None
_GPU_ID = 0


def set_gpu_mode(mode, gpu_id=0):
  """
  Set GPU mode and device ID.

  Args:
    mode (bool): Whether to use the GPU.
    gpu_id (int): GPU ID
  """
  global _GPU_ID
  global _USE_GPU
  global _DEVICE

  _GPU_ID = gpu_id
  _USE_GPU = mode
  _DEVICE = torch.device(('cuda:' + str(_GPU_ID)) if _USE_GPU else 'cpu')


def prefer_gpu():
  """
  Prefer to use GPU(s) if GPU(s) is detected.
  """
  if torch.cuda.is_available():
      set_gpu_mode(True)
  else:
      set_gpu_mode(False)


def global_device():
  """
  Returns the global device that torch.Tensors should be placed on.

  Note: The global device is set by using the function `garage.torch._functions.set_gpu_mode.`
  If this functions is never called `garage.torch._functions.device()` returns None.

  Returns:
    `torch.Device`: The global device that newly created torch.Tensors should be placed on.
  """
  global _DEVICE
  return _DEVICE


def set_seed(seed):
  """
  Set the process-wide random seed.

  Args:
    seed (int): A positive integer
  """
  seed %= 4294967294

  global seed_
  global seed_stream_

  seed_ = seed
  random.seed(seed)
  np.random.seed(seed)

  warnings.warn('Enabeling deterministic mode in PyTorch can have a performance impact when using GPU.')

  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  pass


def get_seed():
  """
  Get the process-wide random seed.

  Returns:
    int: The process-wide random seed
  """
  return seed_

