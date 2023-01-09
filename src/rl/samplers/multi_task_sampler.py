import torch.multiprocessing as mp

from src.rl.samplers import SamplerWorker

class MultiTaskSampler:

  def __init__(self, env_name, env_kwargs, batch_size, policy, baseline, env = None, seed = None, num_workers = 1):
    super(MultiTaskSampler, self).__init__(env_name, env_kwargs, batch_size, policy, seed = seed, env = env)

    self.num_workers = num_workers
    self.task_queue = mp.JoinableQueue()

    self.train_episodes_queue = mp.Queue()
    self.valid_episodes_queue = mp.Queue()

    policy_lock = mp.Lock()

    self.wokers = [
      SamplerWorker
    ]
    pass

  def sample_tasks(self, num_tasks: int):
    return self.env.unwrapped.sample_tasks(num_tasks)

  def sample_async(self, tasks, **kwargs):
    pass

  def sample_wait(self, episodes_futures):
    pass

  def sample(self, tasks, **kwargs):
    pass

  @property
  def train_consumer_thread(self):
    if self._train_consumer_thread is None:
      raise ValueError()

    return self._train_consumer_thread

  @property
  def valid_consumer_thread(self):
    if self._valid_consumer_thread is None:
      raise ValueError()

    return self._valid_consumer_thread

  def _start_consumer_threads(self, tasks, num_steps = 1):
    pass
