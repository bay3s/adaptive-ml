import torch
import torch.nn as nn
from torch.distributions import Normal

from .mlp_module import MLPModule
from .gaussian_mlp_base_module import GaussianMLPBaseModule


class GaussianMLPModule(GaussianMLPBaseModule):
    """
    GaussianMLPModule that mean and std share the same network.

    Args:
      input_dim (int): Input dimension of the model.
      output_dim (int): Output dimension of the model.
      hidden_sizes (list[int]): Output dimension of dense layer(s) for the MLP for mean.
      hidden_nonlinearity (callable): Activation function for intermediate dense layer(s). It should return a
        torch.Tensor. Set it to None to maintain a linear activation.
      hidden_w_init (callable): Initializer function for the weight of intermediate dense layer(s). The function
        should return a torch.Tensor.
      hidden_b_init (callable): Initializer function for the bias of intermediate dense layer(s). The function should
        return a torch.Tensor.
      output_nonlinearity (callable): Activation function for output dense layer. It should return a torch.Tensor. Set
        it to None to maintain a linear activation.
      output_w_init (callable): Initializer function for the weight of output dense layer(s). The function should return
        a torch.Tensor.
      output_b_init (callable): Initializer function for the bias of output dense layer(s). The function should return a
        torch.Tensor.
      learn_std (bool): Is std trainable.
      init_std (float): Initial value for std. (plain value - not log or exponentiated).
      min_std (float): If not None, the std is at least the value of min_std, to avoid numerical issues (plain value -
        not log or exponentiated).
      max_std (float): If not None, the std is at most the value of max_std, to avoid numerical issues (plain value -
        not log or exponentiated).
      std_parameterization (str): How the std should be parametrized. There are two options:
        - exp: the logarithm of the std will be stored, and applied a
           exponential transformation
        - softplus: the std will be computed as log(1+exp(x))
      layer_normalization (bool): Bool for using layer normalization or not.
      normal_distribution_cls (torch.distribution): normal distribution class to be constructed and returned by a call
        to forward. By default, is `torch.distributions.Normal`.
    """

    def __init__(
      self,
      input_dim,
      output_dim,
      hidden_sizes=(32, 32),
      *,
      hidden_nonlinearity=torch.tanh,
      hidden_w_init=nn.init.xavier_uniform_,
      hidden_b_init=nn.init.zeros_,
      output_nonlinearity=None,
      output_w_init=nn.init.xavier_uniform_,
      output_b_init=nn.init.zeros_,
      learn_std=True,
      init_std=1.0,
      min_std=1e-6,
      max_std=None,
      std_parameterization='exp',
      layer_normalization=False,
      normal_distribution_cls=Normal
    ):
      super().__init__(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
        hidden_nonlinearity=hidden_nonlinearity,
        hidden_w_init=hidden_w_init,
        hidden_b_init=hidden_b_init,
        output_nonlinearity=output_nonlinearity,
        output_w_init=output_w_init,
        output_b_init=output_b_init,
        learn_std=learn_std,
        init_std=init_std,
        min_std=min_std,
        max_std=max_std,
        std_parameterization=std_parameterization,
        layer_normalization=layer_normalization,
        normal_distribution_cls=normal_distribution_cls
      )

      self._mean_module = MLPModule(
        input_dim=self._input_dim,
        output_dim=self._action_dim,
        hidden_sizes=self._hidden_sizes,
        hidden_nonlinearity=self._hidden_nonlinearity,
        hidden_w_init=self._hidden_w_init,
        hidden_b_init=self._hidden_b_init,
        output_nonlinearity=self._output_nonlinearity,
        output_w_init=self._output_w_init,
        output_b_init=self._output_b_init,
        layer_normalization=self._layer_normalization
      )
      pass

    # pylint: disable=arguments-differ
    def _get_mean_and_log_std(self, x):
      """
      Get mean and std of Gaussian distribution given inputs.

      Args:
        x: Input to the module.

      Returns:
        torch.Tensor: The mean of Gaussian distribution.
        torch.Tensor: The variance of Gaussian distribution.
      """
      mean = self._mean_module(x)

      return mean, self._log_std
