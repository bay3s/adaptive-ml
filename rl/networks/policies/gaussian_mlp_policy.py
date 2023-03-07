from typing import Tuple
import torch
from torch import nn

from rl.networks.modules import GaussianMLPModule
from rl.networks.policies.stochastic_policy import StochasticPolicy


class GaussianMLPPolicy(StochasticPolicy):
  """
  MLP whose outputs are fed into a Normal distribution.

  A policy that contains a MLP to make prediction based on a gaussian distribution.

  Args:
      env_spec (EnvSpec): Environment specification.
      hidden_sizes (list[int]): Output dimension of dense layer(s) for the MLP for mean.
      hidden_nonlinearity (callable): Activation function for intermediate dense layer(s).
      hidden_w_init (callable): Initializer function for the weight of intermediate dense layer(s).
      hidden_b_init (callable): Initializer function for the bias of intermediate dense layer(s).
      output_nonlinearity (callable): Activation function for output dense layer.
      output_w_init (callable): Initializer function for the weight of output dense layer(s).
      output_b_init (callable): Initializer function for the bias of output dense layer(s).
      learn_std (bool): Is std trainable.
      init_std (float): Initial value for std.
      min_std (float): Minimum value for std.
      max_std (float): Maximum value for std.
      std_parameterization (str): How the std should be parametrized. There are two options:
        - exp: the logarithm of the std will be stored, and applied a exponential transformation
        - softplus: the std will be computed as log(1+exp(x))
      layer_normalization (bool): Bool for using layer normalization or not.
      name (str): Name of policy.
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
    min_std = 1e-6,
    max_std = None,
    std_parameterization = 'exp',
    layer_normalization = False,
    name = 'GaussianMLPPolicy'
  ):
    super().__init__(env_spec, name)
    self._obs_dim = env_spec.observation_space.flat_dim
    self._action_dim = env_spec.action_space.flat_dim

    self._module = GaussianMLPModule(
      input_dim = self._obs_dim,
      output_dim = self._action_dim,
      hidden_sizes = hidden_sizes,
      hidden_nonlinearity = hidden_nonlinearity,
      hidden_w_init = hidden_w_init,
      hidden_b_init = hidden_b_init,
      output_nonlinearity = output_nonlinearity,
      output_w_init = output_w_init,
      output_b_init = output_b_init,
      learn_std = learn_std,
      init_std = init_std,
      min_std = min_std,
      max_std = max_std,
      std_parameterization = std_parameterization,
      layer_normalization = layer_normalization
    )
    pass

  def forward(self, observations: torch.Tensor) -> Tuple:
    """
    Compute the action distributions from the observations.

    Args:
      observations (torch.Tensor): Batch of observations on default torch device.

    Returns:
      torch.distributions.Distribution: Batch distribution of actions.
      dict[str, torch.Tensor]: Additional agent_info, as torch Tensors
    """
    dist = self._module(observations)

    return dist, dict(mean = dist.mean, log_std = (dist.variance ** .5).log())
