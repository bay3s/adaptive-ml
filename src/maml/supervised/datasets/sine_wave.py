import torch


class SineWave:

  def __init__(self, amplitude: float, phase: float):
    """
    Initialize amplitude sine wave

    Args:
      amplitude (int): Amplitude for the Sine wave.
      phase (phase): Phase for the Sine wave function.
    """
    self.amplitude = amplitude
    self.phase = phase

  def sample(self, num_points: int) -> [list, list]:
    """
    Sample a given number of points from the current Sine wave function.

    Args:
      num_points (int): Number of points to sample from the

    Returns:
      Returns an tuple with the x and y coordinates of the sampled points.
    """
    x = torch.rand((num_points, 1)) * 10 - 5
    y = self.amplitude * torch.sin(x + self.phase)

    return x, y


