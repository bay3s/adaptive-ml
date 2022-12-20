import torch.nn as nn
import torch


def _conv_3x3(in_channels, out_channels) -> nn.Sequential:
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size= (3, 3), padding=1),
    nn.BatchNorm2d(out_channels, momentum = 1., track_running_stats = False),
    nn.ReLU(),
    nn.MaxPool2d(2)
  )


class OmniglotCNN(nn.Module):

  def __init__(self, in_channels: int, out_features: int, hidden_size: int = 64):
    """
    Instantiate a CNN for the Omniglot classification task.

    Args:
      in_channels (int): Number of input channels.
      out_features (int): Number of output features.
      hidden_size (int): Size of the hidden layer.
    """
    super(OmniglotCNN, self).__init__()

    self.in_channels = in_channels
    self.out_features = out_features
    self.hidden_size = hidden_size

    self.features = nn.Sequential(
      _conv_3x3(in_channels, hidden_size),
      _conv_3x3(hidden_size, hidden_size),
      _conv_3x3(hidden_size, hidden_size),
      _conv_3x3(hidden_size, hidden_size),
    )

    self.classifier = nn.Linear(hidden_size, out_features)
    pass

  def forward(self, inputs: torch.Tensor) -> torch.Tensor:
    """
    Conducts the forward pass through the network.

    Args:
      inputs (torch.Tensor): Input tensor to the neural net.

    Returns:
      torch.Tensor
    """
    return self.classifier(self.features(inputs))
