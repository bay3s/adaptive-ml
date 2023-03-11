from abc import ABC, abstractmethod


class BaseEnvUpdateHandler(ABC):

  @abstractmethod
  def __call__(self, old_env=None):
    """
    A callable that "updates" an environment.
    """
    raise NotImplementedError
