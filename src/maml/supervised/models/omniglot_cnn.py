import torch.nn as nn

from torchmeta.modules import (
  MetaModule,
  MetaSequential,
  MetaConv2d,
  MetaBatchNorm2d,
  MetaLinear
)


def _conv_3x3(in_channels, out_channels) -> MetaSequential:
  return MetaSequential(
    MetaConv2d(in_channels, out_channels, kernel_size= (3, 3), padding=1),
    MetaBatchNorm2d(out_channels, momentum = 1., track_running_stats = False),
    nn.ReLU(),
    nn.MaxPool2d(2)
  )


class OmniglotCNN(MetaModule):

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

    self.features = MetaSequential(
      _conv_3x3(in_channels, hidden_size),
      _conv_3x3(hidden_size, hidden_size),
      _conv_3x3(hidden_size, hidden_size),
      _conv_3x3(hidden_size, hidden_size),
    )

    self.classifier = MetaLinear(hidden_size, out_features)
    pass

  def forward(self, inputs, params = None):
    features = self.features(inputs, params=self.get_subdict(params, 'features'))
    features = features.view((features.size(0), -1))

    return self.classifier(features, params=self.get_subdict(params, 'classifier'))
