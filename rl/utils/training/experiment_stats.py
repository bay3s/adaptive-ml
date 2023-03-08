from dataclasses import dataclass


@dataclass
class ExperimentStats:
  """
  Stats for an experiment.

  Args:
    total_epochs (int): Total epoches.
    total_iterations (int): Total Iterations.
    total_env_steps (int): Steps collected.
  """
  total_epochs: int
  total_iterations: int
  total_env_steps: int
