"""A value function based on a GaussianMLP model."""
import torch
from torch import nn

from rl.networks.modules import GaussianMLPModule
from .base_value_function import BaseValueFunction


class GaussianMLPValueFunction(BaseValueFunction):
  """
  Gaussian MLP Value Function with Model.
  It fits the input data to a gaussian distribution estimated by
  a MLP.

  Args:
    env_spec (EnvSpec): Environment specification.
    hidden_sizes (list[int]): Output dimension of dense layer(s) for the MLP for mean. For example, (32, 32) means
      the MLP consists of two hidden layers, each with 32 hidden units.
    hidden_nonlinearity (callable): Activation function for intermediate dense layer(s). It should return a
      torch.Tensor. Set it to None to maintain a linear activation.
    hidden_w_init (callable): Initializer function for the weight of intermediate dense layer(s). The function
      should return a torch.Tensor.
    hidden_b_init (callable): Initializer function for the bias of intermediate dense layer(s). The function should return a
      torch.Tensor.
    output_nonlinearity (callable): Activation function for output dense
      layer. It should return a torch.Tensor. Set it to None to
      maintain a linear activation.
    output_w_init (callable): Initializer function for the weight
      of output dense layer(s). The function should return a
      torch.Tensor.
    output_b_init (callable): Initializer function for the bias
      of output dense layer(s). The function should return a
      torch.Tensor.
    learn_std (bool): Is std trainable.
    init_std (float): Initial value for std.
      (plain value - not log or exponentiated).
    layer_normalization (bool): Bool for using layer normalization or not.
    name (str): The name of the value function.
  """

  def __init__(
    self,
    env_spec,
    hidden_sizes = (32, 32),
    hidden_nonlinearity = torch.tanh,
    hidden_w_init = nn.init.xavier_uniform_,
    hidden_b_init = nn.init.zeros_,
    output_nonlinearity = None,
    output_w_init = nn.init.xavier_uniform_,
    output_b_init = nn.init.zeros_,
    learn_std = True,
    init_std = 1.0,
    layer_normalization = False,
    name = 'GaussianMLPValueFunction'
  ):
    super(GaussianMLPValueFunction, self).__init__(env_spec, name)

    input_dim = env_spec.observation_space.flat_dim
    output_dim = 1

    self.module = GaussianMLPModule(
      input_dim = input_dim,
      output_dim = output_dim,
      hidden_sizes = hidden_sizes,
      hidden_nonlinearity = hidden_nonlinearity,
      hidden_w_init = hidden_w_init,
      hidden_b_init = hidden_b_init,
      output_nonlinearity = output_nonlinearity,
      output_w_init = output_w_init,
      output_b_init = output_b_init,
      learn_std = learn_std,
      init_std = init_std,
      min_std = None,
      max_std = None,
      std_parameterization = 'exp',
      layer_normalization = layer_normalization
    )

  def compute_loss(self, obs, returns):
    """
    Compute mean value of loss.

    Args:
      obs (torch.Tensor): Observation from the environment.
      returns (torch.Tensor): Acquired returns with shape.
    Returns:
      torch.Tensor: Calculated negative mean scalar value of objective (float).
    """
    dist = self.module(obs)
    ll = dist.log_prob(returns.reshape(-1, 1))
    loss = -ll.mean()

    return loss

  def forward(self, obs):
    """
    Predict value based on paths.

    Args:
      obs (torch.Tensor): Observation from the environment.

    Returns:
      torch.Tensor: Calculated baselines given observations.
    """
    return self.module(obs).mean.flatten(-2)
