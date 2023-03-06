import enum


class StepType(enum.IntEnum):
  """
  Defines the status of a :class:`~TimeStep` within a sequence.

  * A success sequence terminated at step 4 will look like:
    FIRST, MID, MID, TERMINAL

  * A success sequence terminated at step 5 will look like:
    FIRST, MID, MID, MID, TERMINAL

  * An unsuccessful sequence truncated by time limit will look like:
    FIRST, MID, MID, MID, TIMEOUT
  """

  FIRST = 0
  MID = 1
  TERMINAL = 2
  TIMEOUT = 3

  @classmethod
  def get_step_type(cls, step_cnt: int, max_episode_length: int, done: bool):
    """
    Determines the step type based on step cnt and done signal.

    Args:
        step_cnt (int): current step cnt of the environment.
        max_episode_length (int): maximum episode length.
        done (bool): the done signal returned by Environment.

    Returns:
      StepType: the step type.

    Raises:
      ValueError: if step_cnt is < 1. In this case a environment's `reset()` is likely not called yet and
      the step_cnt is None.
    """
    if step_cnt < 1:
      raise ValueError(f'Expect step_cnt to be >= 1, but got {step_cnt} instead. Did you forget to call `reset()`?')

    if max_episode_length is not None and step_cnt >= max_episode_length:
      return StepType.TIMEOUT
    elif done:
      return StepType.TERMINAL
    elif step_cnt == 1:
      return StepType.FIRST
    else:
      return StepType.MID
