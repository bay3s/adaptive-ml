import torch
from torch.distributions import Normal, Independent


class TanhNormal(torch.distributions.Distribution):

    def __init__(self, loc, scale):
      """
      A distribution induced by applying a tanh transformation to a Gaussian random variable.

      Args:
        loc (torch.Tensor): The mean of this distribution.
        scale (torch.Tensor): The stdev of this distribution.
      """
      self._normal = Independent(Normal(loc, scale), 1)
      super().__init__()
      pass

    def log_prob(self, value, pre_tanh_value=None, epsilon=1e-6):
      """
      The log likelihood of a sample on the Tanh Distribution.

      Args:
        value (torch.Tensor): The sample whose loglikelihood is being computed.
        pre_tanh_value (torch.Tensor): The value prior to having the tanh function applied to it but after it has
          been sampled from the normal distribution.
        epsilon (float): Regularization constant. Making this value larger makes the computation more stable but less
          precise.

      Returns:
        torch.Tensor: The log likelihood of value on the distribution.
      """
      if pre_tanh_value is None:
        pre_tanh_value = torch.log((1 + epsilon + value) / (1 + epsilon - value)) / 2

      norm_lp = self._normal.log_prob(pre_tanh_value)
      ret = (norm_lp - torch.sum(torch.log(self._clip_but_pass_gradient((1. - value**2)) + epsilon), axis=-1))

      return ret

    def sample(self, sample_shape=torch.Size()):
      """
      Return a sample, sampled from this TanhNormal Distribution.

      Args:
        sample_shape (list): Shape of the returned value.

      Note:
        Gradients `do not` pass through this operation.

      Returns:
        torch.Tensor: Sample from this TanhNormal distribution.
      """
      with torch.no_grad():
        return self.rsample(sample_shape=sample_shape)

    def rsample(self, sample_shape=torch.Size()):
      """
      Return a sample, sampled from this TanhNormal Distribution.

      Args:
        sample_shape (list): Shape of the returned value.

      Note:
        Gradients pass through this operation.

      Returns:
        torch.Tensor: Sample from this TanhNormal distribution.
      """
      z = self._normal.rsample(sample_shape)

      return torch.tanh(z)

    def rsample_with_pre_tanh_value(self, sample_shape=torch.Size()):
      """
      Return a sample, sampled from this TanhNormal distribution.

      Args:
        sample_shape (list): shape of the return.

      Note:
        Gradients pass through this operation.

      Returns:
        torch.Tensor: Samples from this distribution.
        torch.Tensor: Samples from the underlying Normal distribution, prior to applying tanh.
      """
      z = self._normal.rsample(sample_shape)

      return z, torch.tanh(z)

    def cdf(self, value):
      """
      Returns the CDF at the value.

      Args:
        value (torch.Tensor): The element where the cdf is being evaluated at.

      Returns:
        torch.Tensor: the result of the cdf being computed.
      """
      return self._normal.cdf(value)

    def icdf(self, value):
      """
      Returns the icdf function evaluated at `value`.

      Args:
        value (torch.Tensor): The element where the cdf is being evaluated at.

      Returns:
        torch.Tensor: the result of the cdf being computed.
      """
      return self._normal.icdf(value)

    @classmethod
    def _from_distribution(cls, new_normal):
      """
      Construct a new TanhNormal distribution from a normal distribution.

      Args:
        new_normal (Independent(Normal)): underlying normal dist for the new TanhNormal distribution.

      Returns:
        TanhNormal: A new distribution whose underlying normal dist is new_normal.
      """
      # pylint: disable=protected-access
      new = cls(torch.zeros(1), torch.zeros(1))
      new._normal = new_normal

      return new

    def expand(self, batch_shape, _instance=None):
      """
      Returns a new TanhNormal distribution.

      Args:
        batch_shape (torch.Size): the desired expanded size.
        _instance(instance): new instance provided by subclasses that need to override `.expand`.

      Returns:
        Instance: New distribution instance with batch dimensions expanded to `batch_size`.
      """
      new_normal = self._normal.expand(batch_shape, _instance)
      new = self._from_distribution(new_normal)

      return new

    def enumerate_support(self, expand=True):
      """
      Returns tensor containing all values supported by a discrete dist.

      Args:
        expand (bool): whether to expand the support over the batch dims to match the distribution's `batch_shape`.

      Note:
        Calls the enumerate_support function of the underlying normal distribution.

      Returns:
        torch.Tensor: Tensor iterating over dimension 0.
      """
      return self._normal.enumerate_support(expand)

    @property
    def mean(self):
      """
      torch.Tensor: mean of the distribution.
      """
      return torch.tanh(self._normal.mean)

    @property
    def variance(self):
      """
      torch.Tensor: variance of the underlying normal distribution.
      """
      return self._normal.variance

    def entropy(self):
      """
      Returns entropy of the underlying normal distribution.

      Returns:
        torch.Tensor: entropy of the underlying normal distribution.
      """
      return self._normal.entropy()

    @staticmethod
    def _clip_but_pass_gradient(x, lower=0., upper=1.):
      """
      Clipping function that allows for gradients to flow through.

      Args:
        x (torch.Tensor): value to be clipped
        lower (float): lower bound of clipping
        upper (float): upper bound of clipping

      Returns:
        torch.Tensor: x clipped between lower and upper.
      """
      clip_up = (x > upper).float()
      clip_low = (x < lower).float()

      with torch.no_grad():
        clip = ((upper - x) * clip_up + (lower - x) * clip_low)

      return x + clip

    def __repr__(self):
      """
      Returns the parameterization of the distribution.

      Returns:
        str: The parameterization of the distribution and underlying distribution.
      """
      return self.__class__.__name__
