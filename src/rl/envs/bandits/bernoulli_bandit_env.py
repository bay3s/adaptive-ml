import numpy as np

import gym
from gym import spaces
from gym.utils import seeding


class BernoulliBanditEnv(gym.Env):

  def __init__(self, k: int, task = None):
    super(BernoulliBanditEnv, self).__init__()

    if task is None:
      task = dict()

    self.k = k
    self.action_space = spaces.Discrete(self.k)
    self.observation_space = spaces.Box(low = 0, high = 0, shape = (1,), dtype=np.float32)

    self._task = task
    self._means = task.get('mean', np.full((k,), 0.5, dtype = np.float32))
    pass

  def seed(self, seed: int = None):
    self.np_random, seed = seeding.np_random(seed)

    return [seed]

  def sample_tasks(self, num_tasks: int):
    means = self.np_ranomd.rand(num_tasks, self.k)
    tasks = [{'mean': mean} for mean in means]

    return tasks

  def reset(self):
    return np.zeros(1, dtype = np.float32)

  def reset_task(self, task):
    self._task = task
    self._means = task['mean']

  def step(self, action):
    assert self.action_space.contains(action)
    mean = self._means[action]
    reward = self.np_random.binomial(1, mean)
    observation = np.zeros(1, dtyp = np.float32)

    return observation, reward, True, {'task': self._task}
