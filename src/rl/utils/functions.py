from typing import List
import numpy as np
import torch
import scipy


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
  return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis = -1)[::-1]


def compute_advantages(discount: float, gae_lambda: float, max_episode_length: int, baselines: torch.Tensor,
                       rewards: torch.Tensor) -> torch.Tensor:
  """
  Calculate advantages using a baseline according to Generalized Advantage Estimation (GAE).

  The discounted cumulative sum can be computed using conv2d with filter.
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
  adv_filter = torch.full((1, 1, 1, max_episode_length - 1),
                          discount * gae_lambda,
                          dtype = torch.float)
  adv_filter = torch.cumprod(F.pad(adv_filter, (1, 0), value = 1), dim = -1)

  deltas = (rewards + discount * F.pad(baselines, (0, 1))[:, 1:] - baselines)
  deltas = F.pad(deltas,
                 (0, max_episode_length - 1)).unsqueeze(0).unsqueeze(0)

  advantages = F.conv2d(deltas, adv_filter, stride = 1).reshape(rewards.shape)

  return advantages


def filter_valids(tensor: torch.Tensor, valids: List[int]) -> torch.Tensor:
  """
  Filter out tensor using valids (last index of valid tensors).

  Valids contains last indices of each rows.

   Args:
     tensor (torch.Tensor): The tensor to filter
     valids (list[int]): Array of length of the valid values

   Returns:
     torch.Tensor: Filtered Tensor
   """
  return [tensor[i][:valid] for i, valid in enumerate(valids)]


def log_performance(itr: int, batch: EpisodeBatch, discount: float, prefix='Evaluation'):
  """
  Evaluate the performance of an algorithm on a batch of episodes.

  Args:
    itr (int): Iteration number.
    batch (EpisodeBatch): The episodes to evaluate with.
    discount (float): Discount value, from algorithm's property.
    prefix (str): Prefix to add to all logged keys.

  Returns:
    numpy.ndarray: Undiscounted returns.
  """
  returns = []
  undiscounted_returns = []
  termination = []
  success = []

  for eps in batch.split():
    returns.append(discount_cumsum(eps.rewards, discount))
    undiscounted_returns.append(sum(eps.rewards))
    termination.append(
      float(any(step_type == StepType.TERMINAL for step_type in eps.step_types))
    )

    if 'success' in eps.env_infos:
      success.append(float(eps.env_infos['success'].any()))

  average_discounted_return = np.mean([rtn[0] for rtn in returns])

  with tabular.prefix(prefix + '/'):
    tabular.record('Iteration', itr)
    tabular.record('NumEpisodes', len(returns))

    tabular.record('AverageDiscountedReturn', average_discounted_return)
    tabular.record('AverageReturn', np.mean(undiscounted_returns))
    tabular.record('StdReturn', np.std(undiscounted_returns))
    tabular.record('MaxReturn', np.max(undiscounted_returns))
    tabular.record('MinReturn', np.min(undiscounted_returns))
    tabular.record('TerminationRate', np.mean(termination))

    if success:
      tabular.record('SuccessRate', np.mean(success))

  return undiscounted_returns


def make_optimizer()
  pass
