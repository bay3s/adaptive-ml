from typing import Tuple

import numpy as np
from gym.envs.mujoco import Walker2dEnv
from gym.utils.ezpickle import EzPickle

from rl.envs.base_meta_env import BaseMetaEnv


class Walker2DRandDirecEnv(BaseMetaEnv, Walker2dEnv, EzPickle):

  def __init__(self):
    """
    Initialize the Walker environment.
    """
    BaseMetaEnv.__init__(self)
    Walker2dEnv.__init__(self)
    EzPickle.__init__(self)

    self.goal_direction = None
    self.set_task(self.sample_tasks(1)[0])
    pass

  def sample_tasks(self, n_tasks: int) -> np.ndarray:
    """
    Sample tasks from the environment.

    Args:
      n_tasks (int): Number of tasks to sample from the environment.

    Returns:
      np.ndarray
    """
    return np.random.choice((-1.0, 1.0), (n_tasks,))

  def set_task(self, task):
    """
    Args:
        task: task of the meta-learning environment
    """
    self.goal_direction = task

  def get_task(self):
    """
    Returns:
        task: task of the meta-learning environment
    """
    return self.goal_direction

  def step(self, a) -> Tuple:
    """
    Take a step in the environment.

    Args:
      a (float): Direction in which to step.

    Returns:
      Tuple
    """
    posbefore = self.sim.data.qpos[0]
    self.do_simulation(a, self.frame_skip)
    posafter, height, ang = self.sim.data.qpos[0:3]
    alive_bonus = 1.0

    reward = (self.goal_direction * (posafter - posbefore) / self.dt)
    reward += alive_bonus
    reward -= 1e-3 * np.square(a).sum()

    done = not (0.8 < height < 2.0 and -1.0 < ang < 1.0)
    ob = self._get_obs()

    return ob, reward, done, {}

  def _get_obs(self) -> np.ndarray:
    """
    Returns an observation in the environment.

    Returns:
      np.ndarray
    """
    qpos = self.sim.data.qpos
    qvel = self.sim.data.qvel

    return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

  def reset_model(self) -> np.ndarray:
    """
    Reset the environment.

    Returns:
      np.ndarray
    """
    self.set_state(
      self.init_qpos + self.np_random.uniform(low = -.005, high = .005, size = self.model.nq),
      self.init_qvel + self.np_random.uniform(low = -.005, high = .005, size = self.model.nv)
    )

    return self._get_obs()

  def viewer_setup(self) -> None:
    """
    Set up the viewer if the mode is set to human.

    Returns:
      None
    """
    self.viewer.cam.trackbodyid = 2
    self.viewer.cam.distance = self.model.stat.extent * 0.5

