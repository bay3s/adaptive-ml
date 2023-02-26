from abc import ABC, abstractmethod


class SequenceLoggerABC(ABC):

  @abstractmethod
  def write_sequence(self, sequence: list) -> None:
    """
    Takes a sequence and logs it based on the implementation of the abstract method.

    Args:
      sequence (list): List to log.

    Returns:
      None
    """
    raise NotImplementedError
