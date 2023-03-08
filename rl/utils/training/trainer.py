import os
import copy
import time
import cloudpickle

from dowel import logger, tabular

from rl.utils.training.train_args import TrainArgs
from rl.utils.training.snapshotter import Snapshotter, SnapshotConfig
from rl.utils.training.experiment_stats import ExperimentStats
from rl.utils.functions.device_functions import set_seed, get_seed


class Trainer:

  def __init__(self, snapshot_config: SnapshotConfig):
    """
    Trainer for RL agents.

    Args:
      snapshot_config (garage.experiment.SnapshotConfig): The snapshot configuration used by Trainer to create the
        snapshotter.
    """
    self._snapshotter = Snapshotter(
      snapshot_config.snapshot_dir,
      snapshot_config.snapshot_mode,
      snapshot_config.snapshot_gap
    )

    self._has_setup = False
    self._plot = False

    self._seed = None
    self._train_args = None
    self._stats = ExperimentStats(
      total_iterations = 0,
      total_env_steps = 0,
      total_epochs = 0,
      last_episode = None
    )

    self._algo = None
    self._env = None
    self._sampler = None
    self._plotter = None

    self._start_time = None
    self._itr_start_time = None
    self.step_itr = None
    self.step_episode = None

    # only used for off-policy algorithms
    self.enable_logging = True

    self._n_workers = None
    self._worker_class = None
    self._worker_args = None

  def setup(self, algo, env):
    """
    Set up trainer for algorithm and environment.

    This method saves algo and env within trainer and creates a sampler.

    Args:
      algo (RLAlgorithm): An algorithm instance. If this algo want to use samplers, it should have a `_sampler` field.
      env (Environment): An environment instance.
    """
    self._algo = algo
    self._env = env

    self._seed = get_seed()

    if hasattr(self._algo, '_sampler'):
      self._sampler = self._algo._sampler

    self._has_setup = True

  def _shutdown_worker(self):
    """
    Shutdown Plotter and Sampler workers.
    """
    if self._sampler is not None:
      self._sampler.shutdown_worker()

  def obtain_episodes(self, itr, agent_update = None, env_update = None):
    """
    Obtain one batch of episodes.

    Args:
      itr (int): Index of iteration (epoch).
      agent_update (object): Value which will be passed into the
          `agent_update_fn` before doing sampling episodes. If a list is
          passed in, it must have length exactly `factory.n_workers`, and
          will be spread across the workers.
      env_update (object): Value which will be passed into the `env_update_fn` before sampling episodes.

    Raises:
      ValueError

    Returns:
      EpisodeBatch: Batch of episodes.
    """
    if self._sampler is None:
      raise ValueError('Could not find `sampler`. the algo should have a `_sampler` field when `setup()` is called')

    if agent_update is None:
      policy = getattr(self._algo, 'exploration_policy', None)
      if policy is None:
        policy = self._algo.policy

      agent_update = policy.get_param_values()

    episodes = self._sampler.obtain_samples(
      itr, self._train_args.batch_size, agent_update = agent_update, env_update = env_update
    )

    self._stats.total_env_steps += sum(episodes.lengths)

    return episodes

  def obtain_samples(self, itr, batch_size = None, agent_update = None, env_update = None):
    """
    Obtain one batch of samples.

    Args:
      itr (int): Index of iteration (epoch).
      batch_size (int): Number of steps in batch.
      agent_update (object): Value which will be passed into the `agent_update_fn` before sampling episodes.
          be spread across the workers.
      env_update (object): Value which will be passed into the `env_update_fn` before sampling episodes.

    Raises:
      ValueError

    Returns:
      list[dict]: One batch of samples.
    """
    eps = self.obtain_episodes(itr, batch_size, agent_update, env_update)

    return eps.to_list()

  def save(self, epoch):
    """
    Save snapshot of current batch.

    Args:
      epoch (int): Epoch.

    Raises:
      NotSetupError
    """
    if not self._has_setup:
      raise ValueError('Use setup() to setup trainer before saving. Setup not complete.')

    logger.log('Saving snapshot...')

    params = dict()
    # Save arguments
    params['seed'] = self._seed
    params['train_args'] = self._train_args
    params['stats'] = self._stats

    # Save states
    params['env'] = self._env
    params['algo'] = self._algo
    params['n_workers'] = self._n_workers
    params['worker_class'] = self._worker_class
    params['worker_args'] = self._worker_args

    self._snapshotter.save_itr_params(epoch, params)

    logger.log('Saved')

  def restore(self, from_dir, from_epoch = 'last'):
    """
    Restore experiment from snapshot.

    Args:
      from_dir (str): Directory of the pickle file to resume experiment from.
      from_epoch (str or int): The epoch to restore from.

    Returns:
      TrainArgs: Arguments for train().
    """
    saved = self._snapshotter.load(from_dir, from_epoch)

    self._seed = saved['seed']
    self._train_args = saved['train_args']
    self._stats = saved['stats']

    set_seed(self._seed)

    self.setup(env = saved['env'], algo = saved['algo'])

    n_epochs = self._train_args.n_epochs
    last_epoch = self._stats.total_epoch
    last_itr = self._stats.total_itr
    total_env_steps = self._stats.total_env_steps
    batch_size = self._train_args.batch_size
    store_episodes = self._train_args.store_episodes
    pause_for_plot = self._train_args.pause_for_plot

    fmt = '{:<20} {:<15}'
    logger.log('Restore from snapshot saved in %s' % self._snapshotter.snapshot_dir)
    logger.log(fmt.format('-- Train Args --', '-- Value --'))
    logger.log(fmt.format('n_epochs', n_epochs))
    logger.log(fmt.format('last_epoch', last_epoch))
    logger.log(fmt.format('batch_size', batch_size))
    logger.log(fmt.format('store_episodes', store_episodes))
    logger.log(fmt.format('pause_for_plot', pause_for_plot))
    logger.log(fmt.format('-- Stats --', '-- Value --'))
    logger.log(fmt.format('last_itr', last_itr))
    logger.log(fmt.format('total_env_steps', total_env_steps))

    self._train_args.start_epoch = last_epoch + 1

    return copy.copy(self._train_args)

  def log_diagnostics(self, pause_for_plot = False):
    """
    Log diagnostics.

    Args:
      pause_for_plot (bool): Pause for plot.
    """
    logger.log('Time %.2f s' % (time.time() - self._start_time))
    logger.log('EpochTime %.2f s' % (time.time() - self._itr_start_time))
    tabular.record('TotalEnvSteps', self._stats.total_env_steps)
    logger.log(tabular)
    pass

  def train(self, n_epochs, batch_size = None, plot = False, store_episodes = False, pause_for_plot = False):
    """
    Start training.

    Args:
      n_epochs (int): Number of epochs.
      batch_size (int or None): Number of environment steps in one batch.
      plot (bool): Visualize an episode from the policy after each epoch.
      store_episodes (bool): Save episodes in snapshot.
      pause_for_plot (bool): Pause for plot.

    Raises:
      NotSetupError: If train() is called before setup().

    Returns:
      float: The average return in last epoch cycle.
    """
    if not self._has_setup:
      raise ValueError('Use setup() to setup trainer before training, setup not complete.')

    # Save arguments for restore
    self._train_args = TrainArgs(
      n_epochs = n_epochs,
      batch_size = batch_size,
      plot = plot,
      store_episodes = store_episodes,
      pause_for_plot = pause_for_plot,
      start_epoch = 0
    )

    log_dir = self._snapshotter.snapshot_dir
    summary_file = os.path.join(log_dir, 'experiment.json')
    dump_json(summary_file, self)

    average_return = self._algo.train(self)
    self._shutdown_worker()

    return average_return

  def step_epochs(self):
    """
    Step through each epoch.

    This function returns a magic generator. When iterated through, this generator automatically performs services such
    as snapshotting and log management. It is used inside train() in each algorithm.

    The generator initializes two variables: `self.step_itr` and `self.step_episode`.

    To use the generator, these two have to be updated manually in each epoch, as the example shows below.

    Yields:
      int: The next training epoch.
    """
    self._start_time = time.time()
    self.step_itr = self._stats.total_itr
    self.step_episode = None

    # Used by integration tests to ensure examples can run one epoch.
    n_epochs = int(
      os.environ.get('GARAGE_EXAMPLE_TEST_N_EPOCHS',
                     self._train_args.n_epochs))

    logger.log('Obtaining samples...')

    for epoch in range(self._train_args.start_epoch, n_epochs):
      self._itr_start_time = time.time()
      with logger.prefix('epoch #%d | ' % epoch):
        yield epoch
        save_episode = (self.step_episode
                        if self._train_args.store_episodes else None)

        self._stats.last_episode = save_episode
        self._stats.total_epoch = epoch
        self._stats.total_itr = self.step_itr

        self.save(epoch)

        if self.enable_logging:
          self.log_diagnostics(self._train_args.pause_for_plot)
          logger.dump_all(self.step_itr)
          tabular.clear()

  def resume(self, n_epochs = None, batch_size = None, plot = None, store_episodes = None, pause_for_plot = None):
    """
    Resume from restored experiment.

    Args:
      n_epochs (int): Number of epochs.
      batch_size (int): Number of environment steps in one batch.
      plot (bool): Visualize an episode from the policy after each epoch.
      store_episodes (bool): Save episodes in snapshot.
      pause_for_plot (bool): Pause for plot.

    Raises:
      NotSetupError: If resume() is called before restore().

    Returns:
      float: The average return in last epoch cycle.
    """
    if self._train_args is None:
      raise ValueError('You must call restore() before resume(), `TrainArgs` are not instantiated.')

    self._train_args.n_epochs = n_epochs or self._train_args.n_epochs
    self._train_args.batch_size = batch_size or self._train_args.batch_size

    if plot is not None:
      self._train_args.plot = plot
    if store_episodes is not None:
      self._train_args.store_episodes = store_episodes
    if pause_for_plot is not None:
      self._train_args.pause_for_plot = pause_for_plot

    average_return = self._algo.train(self)
    self._shutdown_worker()

    return average_return

  def get_env_copy(self):
    """
    Get a copy of the environment.

    Returns:
      Environment: An environment instance.
    """
    if self._env:
      return cloudpickle.loads(cloudpickle.dumps(self._env))

    return None

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
