import torch.nn as nn


class DifferentiableSGD:

  def __init__(self, module: nn.Module, lr: float = 1e-3):
    """
    DifferentiableSGD performs the same optimization step as SGD, but instead
    of updating parameters in-place, it saves updated parameters in new
    tensors, so that the gradient of functions of new parameters can flow back
    to the pre-updated parameters.

    Useful for algorithms such as MAML that needs the gradient of functions of
    post-updated parameters with respect to pre-updated parameters.

    Args:
      module (torch.nn.module): A torch module whose parameters needs to be optimized.
      lr (float): Learning rate of stochastic gradient descent.
    """
    self.module = module
    self.lr = lr
    pass

  def step(self):
    """
    Take an optimization step.
    """
    memo = set()

    def update(module):
      for child in module.children():
        if child not in memo:
          memo.add(child)
          update(child)

      params = list(module.named_parameters())
      for name, param in params:
        # Skip descendant modules' parameters.
        if '.' not in name:
          if param.grad is None:
            continue

          # Original SGD uses param.grad.data
          new_param = param.add(param.grad, alpha = -self.lr)

          del module._parameters[name]
          setattr(module, name, new_param)
          module._parameters[name] = new_param

    update(self.module)
    pass

  def zero_grad(self):
    """
    Sets gradients of all model parameters to zero.
    """
    for param in self.module.parameters():
      if param.grad is not None:
        param.grad.detach_()
        param.grad.zero_()

  def set_grads_none(self):
    """
    Sets gradients for all model parameters to None.

    This is an alternative to `zero_grad` which sets gradients to zero.
    """
    for param in self.module.parameters():
      if param.grad is not None:
        param.grad = None
