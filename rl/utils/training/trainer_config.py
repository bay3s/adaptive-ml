
class TrainerConfig:

  def __init__(self, n_epochs: int, batch_size: int, store_episodes: bool, start_epoch: int):
    """
    Arguments to call train() or resume().

    Args:
      n_epochs (int): Number of epochs.
      batch_size (int): Number of environment steps in one batch.
      store_episodes (bool): Save episodes in snapshot.
      start_epoch (int): The starting epoch, used for when we would like to resume training.
    """
    self.n_epochs = n_epochs
    self.batch_size = batch_size
    self.store_episodes = store_episodes
    self.start_epoch = start_epoch
    pass

