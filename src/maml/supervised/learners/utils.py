import torch
from typing import Any
from collections import OrderedDict
from sklearn.metrics import precision_score


def _compute_accuracy(softmax_output, y_expected: torch.Tensor) -> float:
  """
  Returns the accuracy of the classification outputs.

  Args:
    softmax_output (torch.Tensor): Softmax output of the model.
    y_expected (torch.Tensor): Expected target output.

  Returns:
    float
  """
  with torch.no_grad():
    _, predictions = torch.max(softmax_output, dim = 1)
    accuracy = torch.mean(predictions.eq(y_expected).float())

  return accuracy.item()


def _compute_precision(softmax_output: torch.Tensor, y_expected: torch.Tensor) -> float:
  """
  Returns the accuracy of the classification outputs.

  Args:
    softmax_output (torch.Tensor): Softmax output of the model.
    y_expected (torch.Tensor): Expected target output.

  Returns:
    float
  """
  with torch.no_grad():
    _, predictions = torch.max(softmax_output, dim = 1)
    precision = precision_score(
      y_expected.detach().cpu().numpy(),
      predictions.detach().cpu().numpy(),
      average = 'weighted',
      zero_division = 0
    )

  return precision


def _tensors_to_device(tensors: torch.Tensor, device: str) -> Any:
  """
  Move list / OrderedDict of tensors, or plain torch.Tensor to a specified device.

  Args:
    tensors (Any): A list, OrderedDict, or a plain Tensor
    device (str): The device to move the Tensors onto.

  Returns:
    Any
  """
  if isinstance(tensors, torch.Tensor):
    return tensors.to(device=device)
  elif isinstance(tensors, (list, tuple)):
    return type(tensors)(_tensors_to_device(tensor, device=device) for tensor in tensors)
  elif isinstance(tensors, (dict, OrderedDict)):
    return type(tensors)([(name, _tensors_to_device(tensor, device=device)) for (name, tensor) in tensors.items()])
  else:
    raise ValueError('Unexpected input found in function `_tensors_to_device`')
