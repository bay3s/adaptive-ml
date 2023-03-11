from .base_env import BaseEnv


class Wrapper(BaseEnv):
  """
  A wrapper for an environment that implements the `Environment` API.
  """

  def __init__(self, env):
    """Initializes the wrapper instance.
    Args:
        env (Environment): The environment to wrap
    """
    self._env = env

  def __getattr__(self, name):
    """
    Forward getattr request to wrapped environment.

    Args:
      name (str): attr (str): attribute name

    Returns:
     object: the wrapped attribute.

    Raises:
      AttributeError
    """
    if name.startswith('_'):
      raise AttributeError("attempted to get missing private attribute '{}'".format(name))

    if not hasattr(self._env, name):
      raise AttributeError('Attribute {} is not found'.format(name))

    return getattr(self._env, name)

  @property
  def action_space(self):
    """akro.Space: The action space specification."""
    return self._env.action_space

  @property
  def observation_space(self):
    """akro.Space: The observation space specification."""
    return self._env.observation_space

  @property
  def spec(self):
    """EnvSpec: The environment specification."""
    return self._env.spec

  @property
  def render_modes(self):
    """list: A list of string representing the supported render modes."""
    return self._env.render_modes

  def step(self, action):
    """Step the wrapped env.
    Args:
        action (np.ndarray): An action provided by the agent.
    Returns:
        EnvStep: The environment step resulting from the action.
    """
    return self._env.step(action)

  def reset(self):
    """
    Reset the wrapped env.

    Returns:
      numpy.ndarray: The first observation conforming to `observation_space`.
      dict: The episode-level information.
    """
    return self._env.reset()

  def render(self, mode):
    """
    Render the wrapped environment.

    Args:
      mode (str): the mode to render with.

    Returns:
      object
    """
    return self._env.render(mode)

  def visualize(self):
    """
    Creates a visualization of the wrapped environment.
    """
    self._env.visualize()

  def close(self):
    """
    Close the wrapped env.
    """
    self._env.close()

  @property
  def unwrapped(self):
    """
    garage.Environment: The inner environment.
    """
    return getattr(self._env, 'unwrapped', self._env)
