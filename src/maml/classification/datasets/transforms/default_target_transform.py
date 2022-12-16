from src.maml.classification.datasets.transforms.target_transform import TargetTransform


class DefaultTargetTransform(TargetTransform):

    def __init__(self, class_augmentations):
        super(DefaultTargetTransform, self).__init__()
        self.class_augmentations = class_augmentations

        self._augmentations = dict((augmentation, i + 1)
            for (i, augmentation) in enumerate(class_augmentations))

        self._augmentations[None] = 0

    def __call__(self, target):
        assert isinstance(target, tuple) and len(target) == 2
        label, augmentation = target
        return (label, self._augmentations[augmentation])
