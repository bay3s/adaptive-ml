import warnings
from typing import List, Tuple, Dict
import numpy as np
import torch
from .device_functions import global_device


def np_to_torch(array: np.ndarray) -> torch.Tensor:
  """
  Numpy arrays to PyTorch tensors.

  Args:
    array (np.ndarray): Data in numpy array.

  Returns:
    torch.Tensor: float tensor on the global device.
  """
  tensor = torch.from_numpy(array)

  if tensor.dtype != torch.float32:
    tensor = tensor.float()

  return tensor.to(global_device())


def filter_valids(tensor: torch.Tensor, valids: List[int]) -> torch.Tensor:
  """
  Filter out tensor using valids (last index of valid tensors).

  Valids contains last indices of each row.

   Args:
     tensor (torch.Tensor): The tensor to filter
     valids (list[int]): Array of length of the valid values

   Returns:
     torch.Tensor: Filtered Tensor
   """
  return [tensor[i][:valid] for i, valid in enumerate(valids)]


def unflatten_tensors(flattened, tensor_shapes: List[Tuple]) -> List:
  """
  Unflatten a flattened tensors into a list of tensors.

  Args:
    flattened (np.ndarray): Flattened tensors.
    tensor_shapes (List[Tuple]): Tensor shapes.

  Returns:
    list[np.ndarray]: Unflattened list of tensors.
  """
  tensor_sizes = list(map(np.prod, tensor_shapes))
  indices = np.cumsum(tensor_sizes)[:-1]

  return [
    np.reshape(pair[0], pair[1])
    for pair in zip(np.split(flattened, indices), tensor_shapes)
  ]


def pad_batch_array(array, lengths, max_length = None):
  """
  Convert a packed into a padded array with one more dimension.

  Args:
    array (np.ndarray): Array of length :math:`(N \bullet [T], X^*)`
    lengths (list[int]): List of length :math:`N` containing the length of each episode in the batch array.
    max_length (int): Defaults to max(lengths) if not provided.

  Returns:
    np.ndarray: Of shape :math:`(N, max_length, X^*)`
  """
  assert array.shape[0] == sum(lengths)
  if max_length is None:
    max_length = max(lengths)
  elif max_length < max(lengths):
    # We have at least one episode longther than max_length (whtich is
    # usually max_episode_length).
    # This is probably not a good idea to allow, but RL2 already uses it.
    warnings.warn('Creating a padded array with longer length than requested')
    max_length = max(lengths)

  padded = np.zeros((len(lengths), max_length) + array.shape[1:],
                    dtype = array.dtype)
  start = 0
  for i, length in enumerate(lengths):
    stop = start + length
    padded[i][0:length] = array[start:stop]
    start = stop

  return padded


def stack_tensor_dict_list(tensor_dict_list: List) -> Dict:
  """
  Stack a list of dictionaries of {tensors or dictionary of tensors}.

  Args:
    tensor_dict_list (List): a list of dictionaries of {tensors or dictionary of tensors}.

  Return:
    dict: a dictionary of {stacked tensors or dictionary of stacked tensors}
  """
  keys = list(tensor_dict_list[0].keys())
  ret = dict()
  for k in keys:
    example = tensor_dict_list[0][k]
    dict_list = [x[k] if k in x else [] for x in tensor_dict_list]
    if isinstance(example, dict):
      v = stack_tensor_dict_list(dict_list)
    else:
      v = np.array(dict_list)

    ret[k] = v

  return ret


def concat_tensor_dict_list(tensor_dict_list):
  """
  Concatenate dictionary of list of tensor.

  Args:
    tensor_dict_list (dict[list]): a list of dictionaries of {tensors or dictionary of tensors}.

  Return:
    dict: a dictionary of {stacked tensors or dictionary of stacked tensors}
  """
  keys = list(tensor_dict_list[0].keys())
  ret = dict()

  for k in keys:
    example = tensor_dict_list[0][k]
    dict_list = [x[k] if k in x else [] for x in tensor_dict_list]
    if isinstance(example, dict):
      v = concat_tensor_dict_list(dict_list)
    else:
      v = np.concatenate(dict_list, axis=0)

    ret[k] = v

  return ret


def slice_nested_dict(dict_or_array, start, stop):
  """
  Slice a dictionary containing arrays (or dictionaries). This function is primarily intended for un-batching
  env_infos and action_infos.

  Args:
    dict_or_array (dict[str, dict or np.ndarray] or np.ndarray): A nested dictionary should only contain dictionaries
      and numpy arrays (recursively).
    start (int): First index to be included in the slice.
    stop (int): First index to be excluded from the slice. In other words, these are typical python slice indices.

  Returns:
    dict or np.ndarray: The input, but sliced.
  """
  if isinstance(dict_or_array, dict):
    return {
      k: slice_nested_dict(v, start, stop)
      for (k, v) in dict_or_array.items()
    }

  return dict_or_array[start:stop]


def list_to_tensor(data):
  """
  Convert a list to a PyTorch tensor.

  Args:
    data (list): Data to convert to tensor

  Returns:
    torch.Tensor: A float tensor
  """
  return torch.as_tensor(data, dtype=torch.float32, device=global_device())
