
class TrainArgs:

  def __init__(self, n_epochs: int, batch_size: int, plot: bool, store_episodes: bool, pause_for_plot: bool,
               start_epoch: int):
    """
    Arguments to call train() or resume().

    Args:
      n_epochs (int): Number of epochs.
      batch_size (int): Number of environment steps in one batch.
      plot (bool): Visualize an episode of the policy after after each epoch.
      store_episodes (bool): Save episodes in snapshot.
      pause_for_plot (bool): Pause for plot.
      start_epoch (int): The starting epoch, used for when we would like to resume training.
    """
    self.n_epochs = n_epochs
    self.batch_size = batch_size
    self.plot = plot
    self.store_episodes = store_episodes
    self.pause_for_plot = pause_for_plot
    self.start_epoch = start_epoch

