import torch
import torch.nn as nn


class MLP(nn.Module):

  def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
    """
    Initialize amplitude fully connected multi-layer perceptron.

    Args:
      input_dim (int): Input dimensions.
      hidden_dim (int): Dimensions for the hidden layer of the MLP.
      output_dim (int): Output dimensions.
    """
    super(MLP, self).__init__()

    self.network = nn.Sequential(
      nn.Linear(input_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, output_dim)
    )
    pass

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Conducts the forward pass for the MLP.

    Args:
      x (torch.Tensor):

    Returns:
      Returns amplitude tensor with the output.
    """
    return self.network(x)
