from src.maml.classification.datasets.transforms.target_transform import TargetTransform
from collections import defaultdict
import torch
from typing import Any


class CategoricalTransform(TargetTransform):

  def __init__(self, num_classes: int = None):
    """
    Target transform to return labels in `[0, num_classes)`.

    Parameters
    ----------
    num_classes : int, optional
        Number of classes. If `None`, then the number of classes is inferred
        from the number of individual labels encountered.

    Examples
    --------
    >>> dataset = Omniglot('data', num_classes_per_task=5, meta_train=True)
    >>> task = dataset.sample_task()
    >>> task[0]

    (<PIL.Image.Image image mode=L size=105x105 at 0x11EC797F0>,
    ('images_evaluation/Glagolitic/character12', None))

    >>> dataset = Omniglot('data', num_classes_per_task=5, meta_train=True, target_transform=Categorical(5))
    >>> task = dataset.sample_task()
    >>> task[0]

    (<PIL.Image.Image image mode=L size=105x105 at 0x11ED3F668>, 2)
    """
    super(CategoricalTransform, self).__init__()
    self.num_classes = num_classes
    self._classes = None
    self._labels = None

  def reset(self) -> None:
    """
    Reset the classes and labels for this categorical tranform.

    Returns:
      None
    """
    self._classes = None
    self._labels = None

  @property
  def classes(self):
    """
    Returns classes associated with the current dataset.

    Returns:
      defaultdict
    """
    if self._classes is not None:
      return self._classes

    self._classes = defaultdict(None)
    if self.num_classes is None:
      default_factory: () = lambda: len(self._classes)
    else:
      default_factory: () = lambda: self.labels[len(self._classes)]

    self._classes.default_factory = default_factory

    if (self.num_classes is not None) and (len(self._classes) > self.num_classes):
      raise ValueError('The number of individual labels ({0}) is greater than that defined by `num_classes`')

    return self._classes

  @property
  def labels(self):
    """
    Returns labels.

    Returns:

    """
    if (self._labels is None) and (self.num_classes is not None):
      self._labels = torch.randperm(self.num_classes).tolist()

    return self._labels

  def __call__(self, target: Any):
    """
    Enables you to write classes where the instances behave like functions.

    Args:
      target (Any):

    Returns:
      Returns
    """
    return self.classes[target]

  def __repr__(self) -> str:
    """
    Returns the object representation in string format.

    Returns:
      str
    """
    return '{0}({1})'.format(self.__class__.__name__, self.num_classes or '')
