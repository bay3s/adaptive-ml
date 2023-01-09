import numpy as np
import scipy
import scipy.signal


def shift_advantages_to_positive(advantages: np.ndarray):
  """
  Shifts advantages so that all have positive values after doing so.

  Args:
    advantages (np.ndarray): Array containing the advantages.

  Returns:
    (np.ndarray) shifted such that advantages are positive.
  """
  return (advantages - np.min(advantages)) + 1e-8


def normalize_advantages(advantages: np.ndarray):
  """
  Returns an array containing normalized advantages.

  Args:
    advantages ():

  Returns:
    (np.ndarray)
  """
  return (advantages - np.mean(advantages)) / (advantages.std() + 1e-8)


def discount_cumsum(x, discount):
  """
  From the SciPy docs:
    > The difference-equation filter is called using the command lfilter in SciPy. This command takes as inputs the
    > vector b, the vector a, a signal x and returns the vector y (the same length as x) computed using the equation
    > given above. If  is N-D, then the filter is computed along the axis provided.

  Returns:
    (float):  y[t] - discount*y[t+1] = x[t] or rev(y)[t] - discount*rev(y)[t-1] = rev(x)[t]
  """
  return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis = 0)[::-1]


def explained_variance_1d(y_predicted: np.ndarray, y_expected: np.ndarray) -> float:
  """
  Returns the explained variance given the predicted y values and the expected y values.

  Args:
    y_predicted (np.ndarray): Predicted values of the variable y.
    y_expected (np.ndarray): Expected values for the variable y.
  Returns:
    (float): variance explained by your estimator
  """
  assert y_expected.ndim == 1 and y_predicted.ndim == 1
  variance_y_expected = np.var(y_expected)

  if np.isclose(variance_y_expected, 0):
    if np.var(y_predicted) > 0:
      return 0
    else:
      return 1

  return 1 - np.var(y_expected - y_predicted) / (variance_y_expected + 1e-8)


def set_seed(seed: int):
  """
  Set the random seed for all random number generators

  Args:
      seed (int) : seed to use

  Returns:
      None
  """
  import random
  import tensorflow as tf
  seed %= 4294967294
  random.seed(seed)
  np.random.seed(seed)
  tf.set_random_seed(seed)
  print('using seed %s' % (str(seed)))
  pass

