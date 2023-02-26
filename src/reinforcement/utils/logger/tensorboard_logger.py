import os
import time

from torch.utils.tensorboard import SummaryWriter
from .key_value_logger_abc import KeyValueLoggerABC


class TensorBoardLogger(KeyValueLoggerABC):

  def __init__(self, directory: str, main_tag: str = 'logs'):
    """
    Writes key value pairs to Tensorboard.

    Args:
      directory (str): The directory in which to store the logs.
      main_tag (str): The main tag while logging to Tensorboard.
    """
    os.makedirs(directory, exist_ok = True)
    self._directory = directory
    self._global_step = 1
    self._main_tag = main_tag
    self._writer = SummaryWriter(log_dir = os.path.join(os.path.abspath(directory)))

  def write_key_values(self, key_values: dict) -> None:
    """
    Write key values to Tensorboard

    Args:
      key_values (dict): A dict containing key-value pairs.

    Returns:
      None
    """
    self._writer.add_scalars(
      main_tag = self._main_tag,
      tag_scalar_dict = key_values,
      global_step = self._global_step,
      walltime = time.time()
    )

    self._writer.flush()
    self._global_step += 1

  def close(self) -> None:
    """
    Close the SummaryWriter stream.

    Returns:
      None
    """
    if self._writer:
      self._writer.close()
      self._writer = None
