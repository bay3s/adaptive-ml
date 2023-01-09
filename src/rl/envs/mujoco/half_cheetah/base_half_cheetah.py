from gym.envs.mujoco import HalfCheetahEnv as HalfCheetahEnv_
import numpy as np
from src.rl.utils import logkv



class HalfCheetahEnv(HalfCheetahEnv_):

  def viewer_setup(self) -> None:
    """
    Set the cam distance for observing the environment.

    Returns:
      None
    """
    camera_id = self.model.camera_name2id('track')
    self.viewer.cam.type = 2
    self.viewer.cam.fixedcamid = camera_id
    self.viewer.cam.distance = self.model.stat.extent * 0.35

    # Hide the overlay
    self.viewer._hide_overlay = True

  def render(self, mode = 'human') -> None:
    """
    Render the environment based on mode provided.

    Args:
      mode (str): Renders the environment based on the mode that was provided.

    Returns:
      None
    """
    if mode == 'rgb_array':
      self._get_viewer(mode).render()
      # window size used for old mujoco-py:
      width, height = 500, 500
      data = self._get_viewer(mode).read_pixels(width, height, depth = False)
      return data
    elif mode == 'human':
      self._get_viewer(mode).render()

  def log_diagnostics(self, paths: dict, prefix: str = ''):
    """
    Log diagnostics for the the environment.

    Args:
      paths (dict):
      prefix (str):

    Returns:
      None
    """
    fwrd_vel = [path['env_infos']['forward_vel'] for path in paths]
    final_fwrd_vel = [path['env_infos']['forward_vel'][-1] for path in paths]
    ctrl_cost = [path['env_infos']['reward_ctrl'] for path in paths]

    logkv(prefix + 'AvgForwardVel', np.mean(fwrd_vel))
    logkv(prefix + 'AvgFinalForwardVel', np.mean(final_fwrd_vel))
    logkv(prefix + 'AvgCtrlCost', np.std(ctrl_cost))

    pass
