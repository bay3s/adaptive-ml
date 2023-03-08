import psutil
from rl.utils.functions.device_functions import get_seed
from rl.samplers.workers import DefaultWorker


def identity_function(value):
  """
  Do nothing.

  Args:
    value(object): A value.

  Returns:
    object: The value.
  """
  return value


class WorkerFactory:

  def __init__(self, *, max_episode_length, seed: int = get_seed(), n_workers: int = psutil.cpu_count(logical = False),
               worker_class: type = DefaultWorker, worker_args = None):
    """
    Constructs worers for samplers.

    Args:
      max_episode_length(int): The maximum length episodes which will be sampled.
      seed(int): The seed to use to initialize random number generators.
      n_workers(int): The number of workers to use.
      worker_class(type): Class of the workers.
      worker_args (dict or None): Additional arguments that should be passed to the worker.
    """
    self.n_workers = n_workers
    self._seed = seed
    self._max_episode_length = max_episode_length
    self._worker_class = worker_class

    self._worker_args = {} if worker_args is None else worker_args

  def prepare_worker_messages(self, objs, preprocess = identity_function):
    """
    Take an argument and canonicalize it into a list for all workers.

    Args:
      objs(object or list): Must be either a single object or a list of length n_workers.
      preprocess(function): Function to call on each single object before creating the list.

    Raises:
      ValueError: If a list is passed of a length other than `n_workers`.

    Returns:
      List[object]: A list of length self.n_workers.
    """
    if isinstance(objs, list):
      if len(objs) != self.n_workers:
        raise ValueError('Length of list doesn\'t match number of workers')

      return [preprocess(obj) for obj in objs]

    return [preprocess(objs) for _ in range(self.n_workers)]

  def __call__(self, worker_number):
    """
    Construct a worker given its number.

    Args:
      worker_number(int): The worker number. Should be at least 0 and less than or equal to `n_workers`.

    Raises:
      ValueError: If the worker number is greater than `n_workers`.

    Returns:
      Worker: The constructed worker.
    """

    if worker_number >= self.n_workers:
      raise ValueError('Worker number is too big')

    return self._worker_class(
      worker_number = worker_number,
      seed = self._seed,
      max_episode_length = self._max_episode_length,
      **self._worker_args
    )

