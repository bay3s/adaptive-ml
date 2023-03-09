import os
import pathlib
import json
import time

import dowel
from dowel import logger, tabular

from rl.envs import GymEnv
from rl.samplers.base_sampler import BaseSampler
from rl.utils.modules.log_encoder import LogEncoder
from rl.networks.policies.base_policy import BasePolicy
from rl.utils.training.trainer_config import TrainerConfig
from rl.utils.training.snapshotter import Snapshotter
from rl.utils.training.experiment_stats import ExperimentStats
from rl.utils.functions.device_functions import get_seed
from rl.structs import EpisodeBatch


class Trainer:

  def __init__(self, snapshotter: Snapshotter):
    """
    Trainer for RL agents.

    Args:
      snapshotter (Snapshotter): The snapshotter to be used while training.
    """
    self._snapshotter = snapshotter
    self._has_setup = False

    self._seed = None
    self._config = None
    self._stats = ExperimentStats(
      total_iterations = 0,
      total_env_steps = 0,
      total_epochs = 0
    )

    self._agent = None
    self._env = None

    self._start_time = None
    self._itr_start_time = None
    self.step_itr = None

    # only used for off-policy algorithms
    self.enable_logging = True
    self._n_workers = None
    self._worker_class = None
    self._worker_args = None
    pass

  def setup(self, agent, env: GymEnv) -> None:
    """
    Set up the trainer.

    Args:
      agent (REINFORCE): Policy gradient algorithm instance.
      env (Environment): Environment instance.
    """
    self._agent = agent
    self._env = env
    self._seed = get_seed()
    self._has_setup = True
    pass

  @property
  def _sampler(self) -> BaseSampler:
    """
    Sampler for training the agent.

    Returns:
      BaseSampler
    """
    return self._agent._sampler

  def _shutdown_worker(self):
    """
    Shutdown workers.
    """
    self._sampler.shutdown_workers()

  def obtain_episodes(self, agent_policy: BasePolicy, env_update = None) -> EpisodeBatch:
    """
    Obtain one batch of episodes.

    Args:
      agent_policy (BasePolicy): Policy used for sampling.
      env_update (object): Value which will be passed into the `env_update_fn` before sampling episodes.

    Returns:
      EpisodeBatch
    """
    episodes = self._sampler.obtain_samples(
      num_samples = self._config.batch_size,
      agent_policy = agent_policy,
      env_update = env_update
    )

    self._stats.total_env_steps += sum(episodes.lengths)

    return episodes

  def save_epoch(self, epoch: int):
    """
    Save snapshot of current batch.

    Args:
      epoch (int): Epoch.

    Raises:
      NotSetupError
    """
    if not self._has_setup:
      raise ValueError('Use setup() to setup trainer before saving. Setup not complete.')

    params = dict()
    params['seed'] = self._seed
    params['train_args'] = self._config
    params['stats'] = self._stats
    params['env'] = self._env
    params['algo'] = self._agent
    params['n_workers'] = self._n_workers
    params['worker_class'] = self._worker_class
    params['worker_args'] = self._worker_args

    self._snapshotter.save_snapshot(epoch, params)
    pass

  def train(self, n_epochs: int, batch_size: int, store_episodes: bool = False):
    """
    Start training.

    Args:
      n_epochs (int): Number of epochs.
      batch_size (int): Number of environment steps in one batch.
      store_episodes (bool): Save episodes in snapshot.

    Raises:
      NotSetupError: If train() is called before setup().

    Returns:
      float: The average return in last epoch cycle.
    """
    if not self._has_setup:
      raise ValueError('Use setup() to setup trainer before training, setup not complete.')

    self._config = TrainerConfig(
      n_epochs = n_epochs,
      batch_size = batch_size,
      store_episodes = store_episodes,
      start_epoch = 0
    )

    log_dir = self._snapshotter.snapshot_dir

    summary_file = os.path.join(log_dir, 'experiment_configs.json')
    self.dump_json(summary_file, self)

    progress_file = os.path.join(log_dir, 'experiment_progress.csv')
    logger.add_output(dowel.CsvOutput(progress_file))

    average_return = self._agent.train(self)
    self._shutdown_worker()

    return average_return

  def epochs(self):
    """
    When iterated through, this generator automatically performs snapshotting and log management.

    It is used inside train() in each algorithm.

    To use the generator, these two have to be updated manually in each epoch, as the example shows below.

    Yields:
      int: The next training epoch.
    """
    self._start_time = time.time()
    self.step_itr = self._stats.total_iterations

    for epoch in range(self._config.start_epoch, self._config.n_epochs):
      self._itr_start_time = time.time()

      with logger.prefix('epoch #%d | ' % epoch):
        yield epoch

        # updates experiment stats.
        self._stats.total_epoch = epoch
        self._stats.total_itr = self.step_itr
        self.save_epoch(epoch)

        tabular.record('TimeSinceStart', (time.time() - self._start_time))
        tabular.record('EpochTime', (time.time() - self._itr_start_time))
        tabular.record('TotalEnvSteps', self._stats.total_env_steps)

        logger.log(tabular)
        logger.dump_all(self.step_itr)
        tabular.clear()
        pass

  @property
  def total_env_steps(self):
    """
    Total environment steps collected.

    Returns:
      int: Total environment steps collected.
    """
    return self._stats.total_env_steps

  @total_env_steps.setter
  def total_env_steps(self, value):
    """
    Total environment steps collected.

    Args:
      value (int): Total environment steps collected.
    """
    self._stats.total_env_steps = value

  @staticmethod
  def dump_json(filename: str, data):
    """
    Dump a dictionary to a file in JSON format.

    Args:
      filename(str): Filename for the file.
      data(dict): Data to save to file.
    """
    pathlib.Path(os.path.dirname(filename)).mkdir(parents = True, exist_ok = True)

    with open(filename, 'w') as f:
      json.dump(data, f, indent = 2, sort_keys = False, cls = LogEncoder, check_circular = False)

