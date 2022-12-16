import numpy as np


class MetaDataset(object):
  """Base class for a meta-dataset.
  Parameters
  ----------
  meta_train : bool (default: `False`)
      Use the meta-train split of the dataset. If set to `True`, then the
      arguments `meta_val` and `meta_test` must be set to `False`. Exactly one
      of these three arguments must be set to `True`.
  meta_val : bool (default: `False`)
      Use the meta-validation split of the dataset. If set to `True`, then the
      arguments `meta_train` and `meta_test` must be set to `False`. Exactly one
      of these three arguments must be set to `True`.
  meta_test : bool (default: `False`)
      Use the meta-test split of the dataset. If set to `True`, then the
      arguments `meta_train` and `meta_val` must be set to `False`. Exactly one
      of these three arguments must be set to `True`.
  meta_split : string in {'train', 'val', 'test'}, optional
      Name of the split to use. This overrides the arguments `meta_train`,
      `meta_val` and `meta_test`.
  target_transform : callable, optional
      A function/transform that takes a target, and returns a transformed
      version. See also `torchvision.transforms`.
  dataset_transform : callable, optional
      A function/transform that takes a dataset (ie. a task), and returns a
      transformed version of it. E.g. `transforms.ClassSplitter()`.
  """

  def __init__(self, meta_train = False, meta_val = False, meta_test = False,
               meta_split = None, target_transform = None, dataset_transform = None):
    if meta_train + meta_val + meta_test == 0:
      if meta_split is None:
        raise ValueError('The meta-split is undefined. Use either the '
                         'argument `meta_train=True` (or `meta_val`/`meta_test`), or '
                         'the argument `meta_split="train"` (or "val"/"test").')
      elif meta_split not in ['train', 'val', 'test']:
        raise ValueError('Unknown meta-split name `{0}`. The meta-split '
                         'must be in [`train`, `val`, `test`].'.format(meta_split))
      meta_train = (meta_split == 'train')
      meta_val = (meta_split == 'val')
      meta_test = (meta_split == 'test')
    elif meta_train + meta_val + meta_test > 1:
      raise ValueError('Multiple arguments among `meta_train`, `meta_val` '
                       'and `meta_test` are set to `True`. Exactly one must be set to '
                       '`True`.')
    self.meta_train = meta_train
    self.meta_val = meta_val
    self.meta_test = meta_test
    self._meta_split = meta_split
    self.target_transform = target_transform
    self.dataset_transform = dataset_transform
    self.seed()

  @property
  def meta_split(self):
    if self._meta_split is None:
      if self.meta_train:
        self._meta_split = 'train'
      elif self.meta_val:
        self._meta_split = 'val'
      elif self.meta_test:
        self._meta_split = 'test'
      else:
        raise NotImplementedError()
    return self._meta_split

  def seed(self, seed = None):
    self.np_random = np.random.RandomState(seed = seed)
    # Seed the dataset transform
    _seed_dataset_transform(self.dataset_transform, seed = seed)

  def __iter__(self):
    for index in range(len(self)):
      yield self[index]

  def sample_task(self):
    index = self.np_random.randint(len(self))
    return self[index]

  def __getitem__(self, index):
    raise NotImplementedError()

  def __len__(self):
    raise NotImplementedError()

def _seed_dataset_transform(transform, seed = None):
  if isinstance(transform, Compose):
    for subtransform in transform.transforms:
      _seed_dataset_transform(subtransform, seed = seed)
  elif hasattr(transform, 'seed'):
    transform.seed(seed = seed)
