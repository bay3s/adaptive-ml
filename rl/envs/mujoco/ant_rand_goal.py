from typing import Tuple

import numpy as np
from gym.envs.mujoco import AntEnv
from gym.utils.ezpickle import EzPickle

from rl.envs.base_meta_env import BaseMetaEnv


class AntRandGoalEnv(BaseMetaEnv, AntEnv, EzPickle):

  def __init__(self):
    """
    Initialize the environment.
    """
    self.goal_pos = None
    self.set_task(self.sample_tasks(1)[0])

    BaseMetaEnv.__init__(self)
    AntEnv.__init__(self)
    EzPickle.__init__(self)
    pass

  def sample_tasks(self, n_tasks: int) -> np.ndarray:
    """
    Sample tasks and return a stacked array of said tasks.

    Args:
      n_tasks (int):

    Returns:
      np.ndarray
    """
    a = np.random.random(n_tasks) * 2 * np.pi
    r = 3 * np.random.random(n_tasks) ** 0.5

    return np.stack((r * np.cos(a), r * np.sin(a)), axis = -1)

  def set_task(self, task: float) -> None:
    """
    Set the task in the current environment.

    Args:
      task (float): Task to be set.

    Returns:
      None
    """
    self.goal_pos = task

  def get_task(self):
    """
    Returns:
        task: task of the meta-learning environment
    """
    return self.goal_pos

  def step(self, action: float) -> Tuple:
    """
    Take a step in the environment given the action.

    Args:
      action (float): Action to be taken in the environment.

    Returns:
    Returns:
      Tuple
    """
    self.do_simulation(action, self.frame_skip)
    xposafter = self.get_body_com('torso')
    goal_reward = -np.sum(np.abs(xposafter[:2] - self.goal_pos))
    ctrl_cost = .1 * np.square(action).sum()
    contact_cost = 0.5 * 1e-3 * np.sum(np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))

    # survive_reward = 1.0
    survive_reward = 0.0
    reward = goal_reward - ctrl_cost - contact_cost + survive_reward

    done = False
    ob = self._get_obs()

    return ob, reward, done, dict(
      reward_forward = goal_reward,
      reward_ctrl = -ctrl_cost,
      reward_contact = -contact_cost,
      reward_survive = survive_reward
    )

  def _get_obs(self) -> np.ndarray:
    """
    Get observation from the environment.

    Returns:
      np.ndarray
    """
    return np.concatenate([
      self.sim.data.qpos.flat,
      self.sim.data.qvel.flat,
      np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
    ])

  def reset_model(self) -> np.ndarray:
    """
    Reset the environment.

    Returns:
      np.ndarray
    """
    qpos = self.init_qpos + self.np_random.uniform(size = self.model.nq, low = -.1, high = .1)
    qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
    self.set_state(qpos, qvel)

    return self._get_obs()

  def viewer_setup(self) -> None:
    """
    Set up the viewer for the simulation.

    Returns:
      None
    """
    self.viewer.cam.distance = self.model.stat.extent * 0.5
