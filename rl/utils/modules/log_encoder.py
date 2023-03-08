import enum
import json
import weakref
import numpy as np


class LogEncoder(json.JSONEncoder):

  # a list of modules whose content cannot be jsonified meaningfully.
  BLOCKED_MODULES = {
    'tensorflow',
    'ray',
    'itertools',
  }

  def __init__(self, *args, **kwargs):
    """
    Encoder to be used as cls in json.dump.

    Args:
      args (object): Passed to super class.
      kwargs (dict): Passed to super class.
    """
    super().__init__(*args, **kwargs)
    self._markers = {}

  def default(self, o):
    """
    Perform JSON encoding.

    Args:
      o (object): Object to encode.

    Raises:
      TypeError: If `o` cannot be turned into JSON even using `repr(o)`.

    Returns:
      dict or str or float or bool: Object encoded in JSON.
    """
    if isinstance(o, (int, bool, float, str)):
      return o

    markerid = id(o)
    if markerid in self._markers:
      return 'circular ' + repr(o)

    self._markers[markerid] = o
    try:
      return self._default_inner(o)
    finally:
      del self._markers[markerid]

  def _default_inner(self, o):
    """
    Perform JSON encoding.

    Args:
      o (object): Object to encode.

    Raises:
      TypeError: If `o` cannot be turned into JSON even using `repr(o)`.
      ValueError: If raised by calling repr on an object.

    Returns:
      dict or str or float or bool: Object encoded in JSON.
    """
    try:
      return json.JSONEncoder.default(self, o)
    except TypeError as err:
      if isinstance(o, dict):
        data = {}
        for (k, v) in o.items():
          if isinstance(k, str):
            data[k] = self.default(v)
          else:
            data[repr(k)] = self.default(v)
        return data
      elif isinstance(o, weakref.ref):
        return repr(o)
      elif type(o).__module__.split('.')[0] in self.BLOCKED_MODULES:
        return repr(o)
      elif isinstance(o, type):
        return {'$typename': o.__module__ + '.' + o.__name__}
      elif isinstance(o, np.number):
        return float(o)
      elif isinstance(o, np.bool8):
        return bool(o)
      elif isinstance(o, enum.Enum):
        return {
          '$enum':
            o.__module__ + '.' + o.__class__.__name__ + '.' + o.name
        }
      elif isinstance(o, np.ndarray):
        return repr(o)
      elif hasattr(o, '__dict__') or hasattr(o, '__slots__'):
        obj_dict = getattr(o, '__dict__', None)
        if obj_dict is not None:
          data = {k: self.default(v) for (k, v) in obj_dict.items()}
        else:
          data = {
            s: self.default(getattr(o, s))
            for s in o.__slots__
          }
        t = type(o)
        data['$type'] = t.__module__ + '.' + t.__name__
        return data
      elif callable(o) and hasattr(o, '__name__'):
        if getattr(o, '__module__', None) is not None:
          return {'$function': o.__module__ + '.' + o.__name__}
        else:
          return repr(o)
      else:
        try:
          # This case handles many built-in datatypes like deques
          return [self.default(v) for v in list(o)]
        except TypeError:
          pass
        try:
          # This case handles most other weird objects.
          return repr(o)
        except TypeError:
          pass
        raise err
