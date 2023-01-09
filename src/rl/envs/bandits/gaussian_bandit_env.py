import numpy as np
import gym

from gym import spaces
from gym.utils import seeding


class GaussianBanditEnv(gym.Env):

  def __init__(self, k, std = 1.0, task = {}):
    super(GaussianBanditEnv, self).__init__()
    self.k = k
    self.std = std

    self.action_space = spaces.Discrete(self.k)
    self.observation_space = spaces.Box(low = 0, high = 0, shape = (1,), dtype = np.float32)

    self._task = task
    self._means = task.get('mean', np.full((k,), 0.5, dtype=np.float32))

    self.seed()
    pass

  def seed(self, seed = None):
    self.np_random, seed = seeding.np_random(seed)

    return [seed]

  def sample_tasks(self, num_tasks):
    means = self.np_random.rand(num_tasks, self.k)
    tasks = [{'mean': mean} for mean in means]

    return tasks

  def reset_task(self, task):
    self._task = task
    self._means = task['mean']

  def reset(self):
    return np.zeros(1, dtype = np.float32)

  def step(self, action):
    assert self.action_space.contains(action)
    mean = self._means
    reward = self.np_random.normal(mean, self.std)
    observation = np.zeros(1, dtype = np.float32)

    return observation, reward, True, {'task': self._task}
