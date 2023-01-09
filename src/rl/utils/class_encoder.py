import json


class ClassEncoder(json.JSONEncoder):

  def default(self, o):
    """
    Returns dict with encoded function / object

    Args:
      o (object): Python object or function to encode.

    Returns:
      Encoded value of the function / object.
    """
    if isinstance(o, type):
      return {'$class': o.__module__ + "." + o.__name__}

    if callable(o):
      return {'function': o.__name__}

    return json.JSONEncoder.default(self, o)
