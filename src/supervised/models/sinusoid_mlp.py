import torch
import torch.nn as nn

class SinusoidMLP(nn.Module):

  def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
    """
    Initialize _amplitude fully connected multi-layer perceptron.

    Args:
      input_dim (int): Input dimensions.
      hidden_dim (int): Dimensions for the hidden layer of the MLP.
      output_dim (int): Output dimensions.
    """
    super(SinusoidMLP, self).__init__()

    self.network = nn.Sequential(
      nn.Linear(input_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, output_dim)
    )
    pass

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """
    Conducts the forward pass for the neural net.

    Args:
      inputs (torch.Tensor): Input tensor to the neural net.

    Returns:
      torch.Tensor
    """
    return self.network(inputs)
