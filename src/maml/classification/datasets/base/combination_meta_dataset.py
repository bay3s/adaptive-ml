

class CombinationMetaDataset(MetaDataset):
  """Base class for a meta-dataset, where the classification tasks are over
  multiple classes from a `ClassDataset`.
  Parameters
  ----------
  dataset : `ClassDataset` instance
      A dataset of classes. Each item of `dataset` is a dataset, containing
      all the examples from the same class.
  num_classes_per_task : int
      Number of classes per tasks. This corresponds to `N` in `N-way`
      classification.
  target_transform : callable, optional
      A function/transform that takes a target, and returns a transformed
      version. See also `torchvision.transforms`.
  dataset_transform : callable, optional
      A function/transform that takes a dataset (ie. a task), and returns a
      transformed version of it. E.g. `transforms.ClassSplitter()`.
  """

  def __init__(self, dataset, num_classes_per_task, target_transform = None,
               dataset_transform = None):
    if not isinstance(num_classes_per_task, int):
      raise TypeError('Unknown type for `num_classes_per_task`. Expected '
                      '`int`, got `{0}`.'.format(type(num_classes_per_task)))
    self.dataset = dataset
    self.num_classes_per_task = num_classes_per_task
    # If no target_transform, then use a default target transform that
    # is well behaved for the `default_collate` function (assign class
    # augmentations ot integers).
    if target_transform is None:
      target_transform = DefaultTargetTransform(dataset.class_augmentations)

    super(CombinationMetaDataset, self).__init__(meta_train = dataset.meta_train,
                                                 meta_val = dataset.meta_val, meta_test = dataset.meta_test,
                                                 meta_split = dataset.meta_split, target_transform = target_transform,
                                                 dataset_transform = dataset_transform)

  def __iter__(self):
    num_classes = len(self.dataset)
    for index in combinations(num_classes, self.num_classes_per_task):
      yield self[index]

  def sample_task(self):
    index = self.np_random.choice(len(self.dataset),
                                  size = self.num_classes_per_task, replace = False)
    return self[tuple(index)]

  def __getitem__(self, index):
    if isinstance(index, int):
      raise ValueError('The index of a `CombinationMetaDataset` must be '
                       'a tuple of integers, and not an integer. For example, call '
                       '`dataset[({0})]` to get a task with classes from 0 to {1} '
                       '(got `{2}`).'.format(', '.join([str(idx)
                                                        for idx in range(self.num_classes_per_task)]),
                                             self.num_classes_per_task - 1, index))
    assert len(index) == self.num_classes_per_task
    datasets = [self.dataset[i] for i in index]
    # Use deepcopy on `Categorical` target transforms, to avoid any side
    # effect across tasks.
    task = ConcatTask(datasets, self.num_classes_per_task,
                      target_transform = wrap_transform(self.target_transform,
                                                        self._copy_categorical, transform_type = Categorical))

    if self.dataset_transform is not None:
      task = self.dataset_transform(task)

    return task

  def _copy_categorical(self, transform):
    assert isinstance(transform, Categorical)
    transform.reset()
    if transform.num_classes is None:
      transform.num_classes = self.num_classes_per_task
    return deepcopy(transform)

  def __len__(self):
    num_classes, length = len(self.dataset), 1
    for i in range(1, self.num_classes_per_task + 1):
      length *= (num_classes - i + 1) / i

    if length > sys.maxsize:
      warnings.warn('The number of possible tasks in {0} is '
                    'combinatorially large (equal to C({1}, {2})), and exceeds '
                    'machine precision. Setting the length of the dataset to the '
                    'maximum integer value, which undervalues the actual number of '
                    'possible tasks in the dataset. Therefore the value returned by '
                    '`len(dataset)` should not be trusted as being representative '
                    'of the true number of tasks.'.format(self, len(self.dataset),
                                                          self.num_classes_per_task), UserWarning, stacklevel = 2)
      length = sys.maxsize

    return int(length)
