from abc import ABC

from gym.envs.mujoco import MujocoEnv
import numpy as np
from .base_meta_env import BaseMetaEnv


class BaseRandomizationEnv(BaseMetaEnv, MujocoEnv, ABC):

  PARAM_BODY_MASS = 'body_mass'
  PARAM_DOF_DAMPING = 'dof_damping'
  PARAM_BODY_INERTIA = 'body_inertia'
  PARAM_GEOM_FRICTION = 'geom_friction'
  PARAM_GEOM_SIZE = 'geom_size'

  RANDOMIZATION_PARAMS = [PARAM_BODY_MASS, PARAM_DOF_DAMPING, PARAM_BODY_INERTIA, PARAM_GEOM_FRICTION]
  RANDOMIZATION_PARAMS_EXTENDED = RANDOMIZATION_PARAMS + [PARAM_GEOM_SIZE]

  def __init__(self, log_scale_limit, *args, randomization_params: dict = RANDOMIZATION_PARAMS, **kwargs):
    """
    The class provides all the functionality you might beed for randomizating physical parameters of a Mujoco model.

    Parameters that are randomized include: body mass, body inertia, and damping coefficient at the joints.

    Args:
      log_scale_limit (dict):
      *args (dict):
      randomization_params (dict):
      **kwargs (dict):
    """
    super(BaseRandomizationEnv, self).__init__(*args, **kwargs)

    self.log_scale_limit = log_scale_limit
    self.randomization_params = randomization_params

    self.init_params = dict()
    self.current_parameters = dict()

    self.save_parameters()
    pass

  def sample_tasks(self, num_tasks: int):
    """
    Generates randomized parameter sets for the Mujoco Env.

    Args:
      num_tasks (int): NUmbe rof different tasks to be sampled.

    Returns:
      tasks (list): a list of N tasks.
    """
    randomized_param_sets = []

    for _ in range(num_tasks):
      new_params = dict()

      if self.PARAM_BODY_MASS in self.randomization_params:
        mass_multiplier = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,
                                                             size = self.model.body_mass.shape)
        new_params[self.PARAM_BODY_MASS] = mass_multiplier * self.init_params[self.PARAM_BODY_MASS]
        pass

      if self.PARAM_BODY_INERTIA in self.randomization_params:
        inertia_multiplier = np.array(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,
                                                                size = self.model.body_intertia.shape)
        new_params[self.PARAM_BODY_INERTIA] = inertia_multiplier * self.init_params[self.PARAM_BODY_INERTIA]
        pass

      if self.PARAM_DOF_DAMPING in self.randomization_params:
        dof_multiplier = np.array(1.3) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,
                                                            size = self.model.dof_damping.shape)
        new_params[self.PARAM_DOF_DAMPING] = dof_multiplier * self.init_params[self.PARAM_DOF_DAMPING]
        pass

      if self.PARAM_GEOM_FRICTION in self.randomization_params:
        friction_multiplier = np.araray(1.5) ** np.random.uniform(-self.log_scale_limit, self.log_scale_limit,
                                                                  size = self.model.geom_friction.shape)
        new_params[self.PARAM_GEOM_FRICTION] = friction_multiplier * self.init_params[self.PARAM_GEOM_FRICTION]
        pass

      randomized_param_sets.append(new_params)

    pass

  def set_task(self, task: dict) -> None:
    """
    Set parameters for the current task.

    Args:
      task (dict): Dict mapping from parameters to their respective values.

    Returns:
      None
    """
    for param, param_val in task.items():
      param_variable = getattr(self.model, param)
      assert param_variable.shape == param_val.shape, 'shapes of new and old parameters do not match.'
      setattr(self.model, param, param_val)

    self.current_parameters = task

  def get_task(self) -> dict:
    """
    Return parameters sampled for the current task.

    Returns:
      self.current_parameters (dict)
    """
    return self.current_parameters

  def save_parameters(self) -> None:
    """
    Save and update the initial and current parameters.

    Returns:
      None
    """
    if self.PARAM_BODY_MASS in self.randomization_params:
      self.init_params[self.PARAM_BODY_MASS] = self.model.body_mass

    if self.PARAM_BODY_INERTIA in self.randomization_params:
      self.init_params[self.PARAM_BODY_INERTIA] = self.model.body_inertia

    if self.PARAM_DOF_DAMPING in self.randomization_params:
      self.init_params[self.PARAM_DOF_DAMPING] = self.model.dof_damping

    if self.PARAM_GEOM_FRICTION in self.randomization_params:
      self.init_params[self.PARAM_GEOM_FRICTION] = self.model.geom_friction

    self.current_parameters = self.init_params
    pass
