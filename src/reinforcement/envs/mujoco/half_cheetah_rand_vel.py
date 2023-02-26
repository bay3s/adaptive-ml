from typing import List, Tuple
import numpy as np

from gym.envs.mujoco import HalfCheetahEnv
from gym.utils.ezpickle import EzPickle

from src.reinforcement.envs.base import MetaEnv
from src.reinforcement.utils import logger


class HalfCheetahRandVelEnv(MetaEnv, HalfCheetahEnv, EzPickle):

  def __init__(self):
    """
    Initialize the half cheetah meta environment such that velocity is randomized.
    """
    self.set_task(self.sample_tasks(1)[0])

    MetaEnv.__init__(self)
    HalfCheetahEnv.__init__(self)
    EzPickle.__init__(self)
    pass

  def sample_tasks(self, n_tasks: int) -> np.ndarray:
    """
    Sample tasks for the environment.

    Args:
      n_tasks (int): Number of tasks to sample.

    Returns:
      np.ndarray
    """
    return np.random.uniform(0.0, 3.0, (n_tasks,))

  def set_task(self, task):
    """
    Set the task in the current environment.

    Args:
      task (float): Task to be set.

    Returns:
      None
    """
    self.goal_velocity = task

  def get_task(self):
    """
    Return the task, in this case the goal direction.

    Returns:
      float
    """
    return self.goal_velocity

  def step(self, action):
    """
    Take a step in the environment given the action.

    Args:
      action (float): Action to be taken in the environment.

    Returns:
      Tuple
    """
    xposbefore = self.sim.data.qpos[0]
    self.do_simulation(action, self.frame_skip)
    xposafter = self.sim.data.qpos[0]
    ob = self._get_obs()

    reward_ctrl = - 0.5 * 0.1 * np.square(action).sum()
    forward_vel = (xposafter - xposbefore) / self.dt
    reward_run = - np.abs(forward_vel - self.goal_velocity)

    reward = reward_ctrl + reward_run
    done = False

    return ob, reward, done, dict(forward_vel = forward_vel, reward_run = reward_run, reward_ctrl = reward_ctrl)

  def _get_obs(self) -> np.ndarray:
    """
    Get observation from the environment.

    Returns:
      np.ndarray
    """
    return np.concatenate([
      self.sim.data.qpos.flat[1:],
      self.sim.data.qvel.flat,
    ])

  def reset_model(self) -> np.ndarray:
    """
    Reset the environment and return the observation.

    Returns:
      np.ndarray
    """
    qpos = self.init_qpos + self.np_random.uniform(low = -.1, high = .1, size = self.model.nq)
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

  def log_diagnostics(self, paths: List, prefix: str = '') -> None:
    """
    Log diagnostics for runs in the environment.

    Args:
      paths (List): Paths for which to log diagnostics.
      prefix (str): Prefix for the logs.

    Returns:
      None
    """
    fwrd_vel = [path['env_infos']['forward_vel'] for path in paths]
    final_fwrd_vel = [path['env_infos']['forward_vel'][-1] for path in paths]
    ctrl_cost = [-path['env_infos']['reward_ctrl'] for path in paths]

    logger.logkv(prefix + 'AvgForwardVel', np.mean(fwrd_vel))
    logger.logkv(prefix + 'AvgFinalForwardVel', np.mean(final_fwrd_vel))
    logger.logkv(prefix + 'AvgCtrlCost', np.std(ctrl_cost))
    pass
