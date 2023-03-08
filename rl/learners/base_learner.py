from abc import abstractmethod, ABC
from rl.utils.training.trainer import Trainer


class BaseLearner(ABC):
  """
  Base class for RL algorithms, outlines expected abstract methods.
  """

  @abstractmethod
  def train(self, trainer: Trainer):
    """
    Obtain samples and start actual training for each epoch.

    Args:
        trainer (Trainer): Trainer is passed to give algorithm
            the access to trainer.step_epochs(), which provides services
            such as snapshotting and sampler control.
    """
