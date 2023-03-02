from typing import List, Tuple
import numpy as np
import torch
import scipy
import torch.nn.functional as F

from dowel import tabular


_USE_GPU = False
_DEVICE = None
_GPU_ID = 0


class _Default:  # pylint: disable=too-few-public-methods
  """
  A wrapper class to represent default arguments.

  Args:
    val (object): Argument value.
  """

  def __init__(self, val):
    self.val = val


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


def set_gpu_mode(mode, gpu_id=0):
    """
    Set GPU mode and device ID.

    Args:
      mode (bool): Whether to use the GPU.
      gpu_id (int): GPU ID
    """
    global _GPU_ID
    global _USE_GPU
    global _DEVICE

    _GPU_ID = gpu_id
    _USE_GPU = mode
    _DEVICE = torch.device(('cuda:' + str(_GPU_ID)) if _USE_GPU else 'cpu')


def prefer_gpu():
  """
  Prefer to use GPU(s) if GPU(s) is detected.
  """
  if torch.cuda.is_available():
      set_gpu_mode(True)
  else:
      set_gpu_mode(False)


def global_device():
  """
  Returns the global device that torch.Tensors should be placed on.

  Note: The global device is set by using the function `garage.torch._functions.set_gpu_mode.`
  If this functions is never called `garage.torch._functions.device()` returns None.

  Returns:
    `torch.Device`: The global device that newly created torch.Tensors should be placed on.
  """
  global _DEVICE
  return _DEVICE


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


def make_optimizer(optimizer_type, module=None, **kwargs):
  """
  Create an optimizer for PyTorch.

  Args:
    optimizer_type (Union[type, tuple[type, dict]]): Type of optimizer. This can be an optimizer type such as
      'torch.optim.Adam' or a tuple of type and dictionary, where dictionary contains arguments to initialize the
    optimizer e.g. (torch.optim.Adam, {'lr' : 1e-3})
    module (optional): If the optimizer type is a `torch.optimizer`. The `torch.nn.Module` module whose parameters
      needs to be optimized must be specified.
    kwargs (dict): Other keyword arguments to initialize optimizer. This
      is not used when `optimizer_type` is tuple.

  Returns:
    torch.optim.Optimizer: Constructed optimizer.

  Raises:
    ValueError: Raises value error when `optimizer_type` is tuple, and non-default argument is passed in `kwargs`.
  """
  if isinstance(optimizer_type, tuple):
    opt_type, opt_args = optimizer_type

    for name, arg in kwargs.items():
      if not isinstance(arg, _Default):
        raise ValueError('Should not specify {} and explicit optimizer args at the same time'.format(name))

    if module is not None:
      return opt_type(module.parameters(), **opt_args)

    return opt_type(**opt_args)

  opt_args = {
    k: v.val if isinstance(v, _Default) else v
    for k, v in kwargs.items()
  }

  if module is not None:
    return optimizer_type(module.parameters(), **opt_args)

  return optimizer_type(**opt_args)


def unflatten_tensors(flattened, tensor_shapes: List[Tuple]) -> List:
  """
  Unflatten a flattened tensors into a list of tensors.

  Args:
    flattened (np.ndarray): Flattened tensors.
    tensor_shapes (List[Tuple]): Tensor shapes.

  Returns:
    list[np.ndarray]: Unflattened list of tensors.
  """
  tensor_sizes = list(map(np.prod, tensor_shapes))
  indices = np.cumsum(tensor_sizes)[:-1]

  return [
    np.reshape(pair[0], pair[1])
    for pair in zip(np.split(flattened, indices), tensor_shapes)
  ]
