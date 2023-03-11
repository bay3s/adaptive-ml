import multiprocessing as mp
from multiprocessing import Queue
import queue
from typing import Union, List
import click
import cloudpickle
import setproctitle

from rl.envs.base_env import BaseEnv
from rl.networks.policies.base_policy import BasePolicy

from rl.structs import EpisodeBatch
from rl.samplers.trajectory_samplers.base_sampler import BaseSampler
from rl.samplers.workers.worker_factory import WorkerFactory


class MultiProcessingSampler(BaseSampler):

  def __init__(self, agents: Union[List, BasePolicy], envs: Union[List[BaseEnv], BaseEnv],
               worker_factory: WorkerFactory):
    """
    Sampler that uses multiprocessing to distribute workers.

    Args:
      agents (Policy or List[Policy]): Agent(s) to use to sample episodes.
      envs (Environment or List[Environment]): Environment from which episodes are sampled.
      worker_factory (WorkerFactory): Factory for creating workers.
    """
    self._factory = worker_factory
    self._agents = self._factory.prepare_worker_messages(agents, cloudpickle.dumps)
    self._envs = self._factory.prepare_worker_messages(envs, cloudpickle.dumps)

    self._to_sampler = mp.Queue(2 * self._factory.n_workers)
    self._to_worker = [mp.Queue(1) for _ in range(self._factory.n_workers)]

    for q in self._to_worker:
      q.cancel_join_thread()
      pass

    self._workers = [
      mp.Process(
        target=run_worker,
        kwargs = dict(
          worker_factory = self._factory,
          to_sampler = self._to_sampler,
          to_worker = self._to_worker[worker_number],
          worker_number = worker_number,
          agent = self._agents[worker_number],
          env = self._envs[worker_number],
        ),
        daemon = False
      )
      for worker_number in range(self._factory.n_workers)
    ]

    self._agent_version = 0
    for w in self._workers:
      w.start()

    self.total_env_steps = 0
    pass

  def _push_updates(self, updated_workers, agent_updates, env_updates):
    """
    Apply updates to the workers and (re)start them.

    Args:
      updated_workers (set[int]): Set of workers that don't need to be updated.
      agent_updates (object): Value which will be passed into the `agent_update_fn` before sampling episodes.
      env_updates (object): Value which will be passed into the `env_update_fn` before sampling episodes.
    """
    for worker_number, q in enumerate(self._to_worker):
      if worker_number in updated_workers:
        try:
          q.put_nowait(('continue', ()))
        except queue.Full:
          pass
      else:
        try:
          q.put_nowait(('start', (agent_updates[worker_number], env_updates[worker_number], self._agent_version)))
          updated_workers.add(worker_number)
        except queue.Full:
          pass

  def obtain_samples(self, num_samples: int, agent_policy: BasePolicy, env_update = None) -> EpisodeBatch:
    """
    Collect a given number of transitions.

    Args:
      num_samples (int): Number of transitions to sample.
      agent_policy (BasePolicy): Policy to be used for sampling.
      env_update (EnvUpdateHandler): Value which will be passed into the `env_update_fn`

    Returns:
      EpisodeBatch
    """
    batches = list()
    completed_samples = 0.
    self._agent_version += 1
    updated_workers = set()

    agent_ups = self._factory.prepare_worker_messages(agent_policy, cloudpickle.dumps)
    env_ups = self._factory.prepare_worker_messages(env_update, cloudpickle.dumps)

    with click.progressbar(length = num_samples, label = 'Sampling') as pbar:
      while completed_samples < num_samples:
        self._push_updates(updated_workers, agent_ups, env_ups)

        for _ in range(self._factory.n_workers):
          try:
            tag, contents = self._to_sampler.get_nowait()
            if tag == 'episode':
              batch, version, worker_n = contents
              del worker_n

              if version == self._agent_version:
                batches.append(batch)
                num_returned_samples = batch.lengths.sum()
                completed_samples += num_returned_samples
                pbar.update(num_returned_samples)
            else:
              raise AssertionError('Unknown tag {} with contents {}'.format(tag, contents))
          except queue.Empty:
            pass

      for q in self._to_worker:
        try:
          q.put_nowait(('stop', ()))
        except queue.Full:
          pass

    samples = EpisodeBatch.concatenate(*batches)
    self.total_env_steps += sum(samples.lengths)

    return samples

  def shutdown_workers(self) -> None:
    """
    Shutdown the workers.

    Returns:
      None
    """
    for (q, w) in zip(self._to_worker, self._workers):
      while True:
        try:
          q.put(('exit', ()), timeout = 1)
          break
        except queue.Full:
          if not w.is_alive():
            break
      w.join()

    for q in self._to_worker:
      q.close()

    self._to_sampler.close()
    pass

  @classmethod
  def from_worker_factory(cls, worker_factory: WorkerFactory, agents: Union[BasePolicy, List], envs: List[BaseEnv]):
    """
    Construct this sampler.

    Args:
      worker_factory (WorkerFactory): Pickleable factory for creating workers.
      agents (Policy or List[Policy]): Agent(s) to use to sample episodes.
      envs(Environment or List[Environment]): Environment from which episodes are sampled.

    Returns:
      Sampler: An instance of `cls`.
    """
    return cls(agents, envs, worker_factory = worker_factory)


def run_worker(worker_factory: WorkerFactory, to_worker: Queue, to_sampler: Queue, worker_number: int,
               agent: BasePolicy, env: BaseEnv):
  """
  Run the streaming worker state machine.

  Args:
    worker_factory (WorkerFactory): Pickleable factory for creating workers.
    to_worker (multiprocessing.Queue): Queue to send commands to the worker.
    to_sampler (multiprocessing.Queue): Queue to send episodes back to the sampler.
    worker_number (int): Number of this worker.
    agent (BasePolicy): Agent to use to sample episodes.
    env (BaseEnv): Environment from which episodes are sampled.

  Raises:
    AssertionError
  """
  to_sampler.cancel_join_thread()
  setproctitle.setproctitle('worker: ' + setproctitle.getproctitle())

  inner_worker = worker_factory(worker_number)
  inner_worker.update_agent(cloudpickle.loads(agent))
  inner_worker.update_env(cloudpickle.loads(env))

  version = 0
  streaming_samples = False

  while True:
    if streaming_samples:
      try:
        tag, contents = to_worker.get_nowait()
      except queue.Empty:
        tag = 'continue'
        contents = None
    else:
      tag, contents = to_worker.get()

    if tag == 'start':
      agent_update, env_update, version = contents
      inner_worker.update_agent(cloudpickle.loads(agent_update))
      inner_worker.update_env(cloudpickle.loads(env_update))
      streaming_samples = True

    elif tag == 'stop':
      streaming_samples = False

    elif tag == 'continue':
      batch = inner_worker.rollout()

      try:
        to_sampler.put_nowait(('episode', (batch, version, worker_number)))
      except queue.Full:
        streaming_samples = False

    elif tag == 'exit':
      to_worker.close()
      to_sampler.close()
      inner_worker.shutdown()
      return

    else:
      raise AssertionError('Unknown tag {} with contents {}'.format(tag, contents))
