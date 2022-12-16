class TargetTransform(object):

  def __call__(self, target):
    raise NotImplementedError()

  def __repr__(self):
    return str(self.__class__.__name__)
