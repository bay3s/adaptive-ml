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


def update_module_params(module, new_params):
  """
  Load parameters to a module.

  Args:
    module (torch.nn.Module): A torch module.
    new_params (dict): A dict of torch tensor used as the new parameters of this module.
  """
  named_modules = dict(module.named_modules())

  def update(m, name, param):
    del m._parameters[name]  # noqa: E501
    setattr(m, name, param)
    m._parameters[name] = param  # noqa: E501

  for name, new_param in new_params.items():
    if '.' in name:
      module_name, param_name = tuple(name.rsplit('.', 1))
      if module_name in named_modules:
        update(named_modules[module_name], param_name, new_param)
    else:
      update(module, name, new_param)
