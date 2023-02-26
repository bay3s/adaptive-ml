from abc import ABC, abstractmethod


class KeyValueLoggerABC(ABC):

  @abstractmethod
  def write_key_values(self, key_value_store: dict) -> None:
    """
    Takes a key-value store (dict) and logs it in the appropriate format.

    Args:
      key_value_store (dict): The key-value store to log.

    Returns:
      (None)
    """
    raise NotImplementedError
