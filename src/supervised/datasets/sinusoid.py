import torch

class Sinusoid:

  def __init__(self, amplitude: float, phase: float):
    """
    Initialize _amplitude sine wave

    Args:
      amplitude (int): Amplitude for the Sine wave.
      phase (int): Phase for the Sine wave function.
    """
    self._amplitude = amplitude
    self._phase = phase

  def sample(self, num_points: int) -> [list, list]:
    """
    Sample a given number of points from the current Sinusoid.

    Args:
      num_points (int): Number of points to sample from the Sinusoid.

    Returns:
      Returns an tuple with the x and y coordinates of the sampled points.
    """
    x = torch.rand((num_points, 1)) * 10 - 5
    y = self._amplitude * torch.sin(x + self._phase)

    return x.tolist(), y.tolist()
