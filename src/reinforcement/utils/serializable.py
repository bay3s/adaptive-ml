import inspect
import sys
from typing import Any

class Serializable(object):

  def __init__(self, *args: Any, **kwargs: Any):
    """
    Initialize the Serializable class.

    Args:
      *args (Any): Arguments for initializing the class.
      **kwargs (Any): Keyword arguments for initializing the class.
    """
    self.__args = args
    self.__kwargs = kwargs
    pass

  def quick_init(self, locals_: Any) -> None:
    """
    Quick init.

    Args:
      locals_ (Any): Local

    Returns:
      None
    """
    try:
      if object.__getattribute__(self, '_serializable_initialized'):
        return
    except AttributeError:
      pass

    if sys.version_info >= (3, 0):
      spec = inspect.getfullargspec(self.__init__)

      if spec.varkw:
        kwargs = locals_[spec.varkw]
      else:
        kwargs = dict()

    else:
      spec = inspect.getargspec(self.__init__)

      if spec.keywords:
        kwargs = locals_[spec.keywords]
      else:
        kwargs = dict()

    if spec.varargs:
      varargs = locals_[spec.varargs]
    else:
      varargs = tuple()

    in_order_args = [locals_[arg] for arg in spec.args][1:]
    self.__args = tuple(in_order_args) + varargs
    self.__kwargs = kwargs
    setattr(self, '_serializable_initialized', True)
    pass

  def __getstate__(self) -> dict:
    """
    Get state of the object.

    Returns:
      dict
    """
    return {'__args': self.__args, '__kwargs': self.__kwargs}

  def __setstate__(self, d) -> None:
    out = type(self)(*d['__args'], **d['__kwargs'])
    self.__dict__.update(out.__dict__)

  @classmethod
  def clone(cls, obj, **kwargs) -> object:
    """
    Clone a Serializable object.

    Args:
      obj (object): Serializable object to clone.
      **kwargs (Any): Key word arguments for cloning the object.

    Returns:
      object
    """
    assert isinstance(obj, Serializable)
    d = obj.__getstate__()

    # Split the entries in kwargs between positional and keyword arguments
    # and update d['__args'] and d['__kwargs'], respectively.
    if sys.version_info >= (3, 0):
      spec = inspect.getfullargspec(obj.__init__)
    else:
      spec = inspect.getargspec(obj.__init__)
    in_order_args = spec.args[1:]

    d['__args'] = list(d['__args'])
    for kw, val in kwargs.items():
      if kw in in_order_args:
        d['__args'][in_order_args.index(kw)] = val
      else:
        d['__kwargs'][kw] = val

    out = type(obj).__new__(type(obj))
    out.__setstate__(d)

    return out
