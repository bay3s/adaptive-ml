import akro
import numpy as np
import torch.nn.functional as F
import torch
from scipy import signal


def discount_cumsum(x: np.ndarray, discount: float):
  """
  Discounted cumulative sum.

  See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering  # noqa: E501

  Here, we have y[t] - discount*y[t+1] = x[t] or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]

  Args:
    x (np.ndarrary): Input.
    discount (float): Discount factor.

  Returns:
    np.ndarrary: Discounted cumulative sum.
  """
  return signal.lfilter([1], [1, float(-discount)], x[::-1], axis = -1)[::-1]


def compute_advantages(discount: float, gae_lambda: float, max_episode_length: int, baselines: torch.Tensor,
                       rewards: torch.Tensor) -> torch.Tensor:
  """
  Calculate advantages using a baseline according to Generalized Advantage Estimation (GAE).

  The discounted cumulative sum can be computed using Conv2D with filter.
    [1, (discount * gae_lambda), (discount * gae_lambda) ^ 2, ...]
    where the length is same with max_episode_length.

  baselines and rewards are also has same shape.
    [ [b_11, b_12, b_13, ... b_1n],
      [b_21, b_22, b_23, ... b_2n],
      ...
      [b_m1, b_m2, b_m3, ... b_mn] ]
    rewards:
    [ [r_11, r_12, r_13, ... r_1n],
      [r_21, r_22, r_23, ... r_2n],
      ...
      [r_m1, r_m2, r_m3, ... r_mn] ]

  Args:
    discount (float): RL discount factor (i.e. gamma).
    gae_lambda (float): Lambda, as used for Generalized Advantage Estimation (GAE).
    max_episode_length (int): Maximum length of a single episode.
    baselines (torch.Tensor): A 2D vector of value function estimates with shape (N, T), where N is the batch
      dimension (number of episodes) and T is the maximum episode length experienced by the agent. If an episode
      terminates in fewer than T time steps, the remaining elements in that episode should be set to 0.
    rewards (torch.Tensor): A 2D vector of per-step rewards with shape (N, T), where N is the batch dimension
      (number of episodes) and T is the maximum episode length experienced by the agent. If an episode terminates in
      fewer than T time steps, the remaining elements in that episode should be set to 0.

  Returns:
    torch.Tensor: A 2D vector of calculated advantage values with shape (N, T), where N is the batch dimension
    (number of episodes) and T is the maximum episode length experienced by the agent. If an episode terminates in
    fewer than T time steps, the remaining values in that episode should be set to 0.
  """
  adv_filter = torch.full((1, 1, 1, max_episode_length - 1), discount * gae_lambda, dtype = torch.float)
  adv_filter = torch.cumprod(F.pad(adv_filter, (1, 0), value = 1), dim = -1)

  deltas = (rewards + discount * F.pad(baselines, (0, 1))[:, 1:] - baselines)
  deltas = F.pad(deltas, (0, max_episode_length - 1)).unsqueeze(0).unsqueeze(0)

  advantages = F.conv2d(deltas, adv_filter, stride = 1).reshape(rewards.shape)

  return advantages


def space_soft_contains(space: akro.Space, element: object) -> bool:
  """
  Check that a space has the same dimensionality as an element. If the space's dimensionality is not available,
  check that the space contains the element.

  Args:
    space (akro.Space or gym.Space): Space to check
    element (object): Element to check in space.

  Returns:
    bool
  """
  if space.contains(element):
    return True
  elif hasattr(space, 'flat_dim'):
    return space.flat_dim == np.prod(element.shape)

  return False
