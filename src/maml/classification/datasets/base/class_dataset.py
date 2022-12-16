import warnings
from ordered_set import OrderedSet
from torchvision.transforms import Compose
from src.maml.classification.datasets.transforms.fixed_category_transform import FixedCategoryTransform


class ClassDataset:

  def __init__(self, meta_split: str, transforms: list = None):
    """
    Base class for a dataset of classes where each item is a dataset containing instances from the same class.

    Args:
      meta_split (str): One of 'train', 'test', 'validation'.
      transforms (list): A list of augmentations to be applied to the current dataset.

    Returns:
      None
    """
    self.meta_split = meta_split
    self.augmentations = self._unique_augmentations(transforms)
    pass

  @staticmethod
  def _unique_augmentations(augmentations: list) -> list:
    """
    Checks if the augmentations provided are unique, if not throws an exception.

    Args:
      augmentations (list): A list of augmentations to be applied to the dataset.

    Returns:

    """
    if not isinstance(augmentations, list):
      raise TypeError('Expected `list` as input for `augmentations`.')

    unique_augmentations = OrderedSet()
    for a in augmentations:
      if a in unique_augmentations:
        warnings.warn('Ignored duplicate augmentation provided for the dataset.', stacklevel = 2)

      unique_augmentations.add(a)
      continue

    return list(unique_augmentations)

  def get_class_augmentation(self, index):
    """
    Returns the augmentation associated with a specific index.

    Args:
      index (int): INdex for the class augmentation.

    Returns:

    """
    transform_index = (index // self.num_classes) - 1

    if transform_index < 0:
      return None

    return self.augmentations[index]

  def get_transform(self, index: int, transform = None) -> Compose:
    """

    Args:
      index (int):
      transform ():

    Returns:
      Compose
    """
    class_transform = self.get_class_augmentation(index)

    if class_transform is None:
      return transform

    if transform is None:
      return class_transform

    return Compose([class_transform, transform])

  def get_target_transform(self, index):
    t = self.get_class_augmentation(index)

    return FixedCategoryTransform(t)

  @property
  def num_classes(self) -> int:
    """
    Returns the number of classes in the dataset.

    Returns:
      int
    """
    return NotImplementedError()

  def __len__(self):
    """
    Returns the length of the dataset

    Returns:
      int
    """
    return self.num_classes * (len(self.augmentations) + 1)
