import numpy as np
from collections import defaultdict
from dowel import tabular

from rl.structs import EpisodeBatch, StepType
from rl.utils.functions.rl_functions import discount_cumsum


def log_performance(training_iteration: int, batch: EpisodeBatch, discount: float, prefix = 'Evaluation'):
  """
  Evaluate the performance of an algorithm on a batch of episodes.

  Args:
    training_iteration (int): Iteration number.
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
  std_discounted_return = np.std([rtn[0] for rtn in returns])

  with tabular.prefix(prefix + '/'):
    tabular.record('Iteration', training_iteration)
    tabular.record('NumEpisodes', len(returns))
    tabular.record('MeanDiscountedReturn', average_discounted_return)
    tabular.record('StdDiscountedReturn', std_discounted_return)
    tabular.record('MeanUndiscountedReturn', np.mean(undiscounted_returns))
    tabular.record('StdUndiscountedReturn', np.std(undiscounted_returns))
    tabular.record('MaxReturn', np.max(undiscounted_returns))
    tabular.record('MinReturn', np.min(undiscounted_returns))
    tabular.record('TerminationRate', np.mean(termination))

    if success:
      tabular.record('SuccessRate', np.mean(success))

  return undiscounted_returns


def log_multitask_performance(itr, batch, discount, name_map = None):
  r"""Log performance of episodes from multiple tasks.
  Args:
      itr (int): Iteration number to be logged.
      batch (EpisodeBatch): Batch of episodes. The episodes should have
          either the "task_name" or "task_id" `env_infos`. If the "task_name"
          is not present, then `name_map` is required, and should map from
          task id's to task names.
      discount (float): Discount used in computing returns.
      name_map (dict[int, str] or None): Mapping from task id's to task
          names. Optional if the "task_name" environment info is present.
          Note that if provided, all tasks listed in this map will be logged,
          even if there are no episodes present for them.
  Returns:
      numpy.ndarray: Undiscounted returns averaged across all tasks. Has
          shape :math:`(N \bullet [T])`.
  """
  eps_by_name = defaultdict(list)
  for eps in batch.split():
    task_name = '__unnamed_task__'
    if 'task_name' in eps.env_infos:
      task_name = eps.env_infos['task_name'][0]
    elif 'task_id' in eps.env_infos:
      name_map = {} if name_map is None else name_map
      task_id = eps.env_infos['task_id'][0]
      task_name = name_map.get(task_id, 'Task #{}'.format(task_id))
    eps_by_name[task_name].append(eps)
  if name_map is None:
    task_names = eps_by_name.keys()
  else:
    task_names = name_map.values()
  for task_name in task_names:
    if task_name in eps_by_name:
      episodes = eps_by_name[task_name]
      log_performance(
        itr,
        EpisodeBatch.concatenate(*episodes),
        discount,
        prefix = task_name
      )
    else:
      with tabular.prefix(task_name + '/'):
        tabular.record('Iteration', itr)
        tabular.record('NumEpisodes', 0)
        tabular.record('AverageDiscountedReturn', np.nan)
        tabular.record('AverageReturn', np.nan)
        tabular.record('StdReturn', np.nan)
        tabular.record('MaxReturn', np.nan)
        tabular.record('MinReturn', np.nan)
        tabular.record('TerminationRate', np.nan)
        tabular.record('SuccessRate', np.nan)

  return log_performance(itr, batch, discount = discount, prefix = 'Average')
