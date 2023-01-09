import numpy as np
import gym

from gym import spaces
from gym.utils import seeding


class TabularMDPEnv(gym.Env):

  def __init__(self, num_states: int, num_actions: int, task: dict = dict()):
    super(TabularMDPEnv, self).__init__()

    self.num_states = num_states
    self.num_actions = num_actions

    self.action_space = spaces.Discrete(num_actions)
    self.observation_space = spaces.Box(low = 0., high = 1., shape = (num_states, ), dtype = np.float32)

    self._task = task
    self._transitions = task.get(
      'transitions',
      np.full((num_states, num_actions, num_states), 1.0 / num_states, dtype = np.float32)
    )

    self._rewards_mean = task.get(
      'rewards_mean',
      np.zeros((num_states, num_actions), dtype = np.float32)
    )

    self._state = 0

    # @todo update the variable name here.
    self.np_random = None
    self.seed()
    pass

  def seed(self, seed: int = None) -> list:
    self.np_random, seed = seeding.np_random(seed)

    return [seed]

  def sample_tasks(self, num_tasks: int):
    transitions = self.np_random.dirichlet(
      alpha = np.ones(self.num_states),
      size = (num_tasks, self.num_states, self.num_actions)
    )

    rewards_mean = self.np_random.normal(1., 1., size = (num_tasks, self.num_states, self.num_actions))

    tasks = [{'transitions': transition, 'rewards_mean': reward_mean}
             for (transition, reward_mean) in zip(transitions, rewards_mean)]

    return tasks

  def reset_task(self, task) -> None:
    self._task = task
    self._transitions = task['transitions']
    self._rewards_mean = task['rewards_mean']
    pass

  def reset(self) -> np.array:
    """
    Resets the environment to an initial state and returns an initial observation.

    Note that this function should not reset the environment's random
    number generator(s); random variables in the environment's state should
    be sampled independently between multiple calls to `reset()`. In other
    words, each call of `reset()` should yield an environment suitable for
    a new episode, independent of previous episodes.

    Returns:
       observation (object): the initial observation.
    """
    self._state = 0
    observation = np.zeros(self.num_states, dtype = np.float32)
    observation[self._state] = 1.

    return observation

  def step(self, action: int):
    """
    Run one timestep of the environment's dynamics. When end of
    episode is reached, you are responsible for calling `reset()`
    to reset this environment's state.

    Accepts an action and returns a tuple (observation, reward, done, info).

    Args:
        action (object): an action provided by the agent

    Returns:
        observation (object): agent's observation of the current environment
        reward (float) : amount of reward returned after previous action
        done (bool): whether the episode has ended, in which case further step() calls will return undefined results
        info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
    """
    assert self.action_space.contains(action)

    mean = self._rewards_mean[self._state, action]
    reward = self.np_random.normal(mean, 1.)

    self._state = self.np_random.choice(self.num_states, p = self._transitions[self._state, action])

    observation = np.zeros(self.num_states, dtype = np.float32)
    observation[self._state] = 1.

    return observation, reward, False, {'task': self._task}
