from torch.utils.data import Dataset as Dataset_
from torchvision.transforms import Compose


class Dataset(Dataset_):

  def __init__(self, index, transform = None, target_transform = None):
    self.index = index
    self.transform = transform
    self.target_transform = target_transform

  def target_transform_append(self, transform):
    if transform is None:
      return
    if self.target_transform is None:
      self.target_transform = transform
    else:
      self.target_transform = Compose([self.target_transform, transform])

  def __hash__(self):
    return hash(self.index)
