from .rl_functions import discount_cumsum
import numpy as np
from dowel import tabular
from rl.structs import EpisodeBatch, StepType


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

