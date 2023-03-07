from abc import abstractmethod, ABC
from rl.trainers import BaseTrainer


class BaseLearner(ABC):
  """
  Base class for RL algorithms, outlines expected abstract methods.
  """

  @abstractmethod
  def train(self, trainer: BaseTrainer):
    """
    Obtain samples and start actual training for each epoch.

    Args:
        trainer (Trainer): Trainer is passed to give algorithm
            the access to trainer.step_epochs(), which provides services
            such as snapshotting and sampler control.
    """
