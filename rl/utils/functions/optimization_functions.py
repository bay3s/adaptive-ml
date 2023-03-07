import torch


def zero_optim_grads(optim: torch.optim.Optimizer, set_to_none: bool = True) -> None:
  """
  Sets the gradient of all optimized tensors to None.
  This is an optimization alternative to calling `optimizer.zero_grad()`

  Args:
    optim (torch.nn.Optimizer): The optimizer instance to zero parameter gradients.
    set_to_none (bool): Set gradients to None instead of calling `zero_grad()`which sets to 0.

  Returns:
    None
  """
  if not set_to_none:
    optim.zero_grad()
    return

  for group in optim.param_groups:
    for param in group['params']:
      param.grad = None
