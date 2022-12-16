
class FixedCategoryTransform(object):

  def __init__(self, transform = None):
    """
    Initialize the class.

    Args:
      transform ():
    """
    self.transform = transform

  def __call__(self, index):
    """


    Args:
      index (): Index to

    Returns:
      Returns a tuple containing the transform and the index.
    """
    return (index, self.transform)

  def __repr__(self):
    """
    Returns a string representation for the object of the class.

    Returns:
      str
    """
    return ('{0}({1})'.format(self.__class__.__name__, self.transform))
