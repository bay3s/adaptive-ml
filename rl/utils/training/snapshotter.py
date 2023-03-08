import errno
import os
import pathlib

import cloudpickle


class Snapshotter:

  def __init__(self, snapshot_dir = os.path.join(os.getcwd(), 'data/local/experiment'), snapshot_mode = 'last',
               snapshot_gap: int = 1):
    """
    Takes snapshot of the state of the

    Args:
      snapshot_dir (str): Path to save the log and iteration snapshot.
      snapshot_mode (str): Mode to save the snapshot.
        - "all" (all iterations will be saved)
        - "last" (only the last iteration will be saved)
        - "gap" (every snapshot_gap iterations are saved)
        - "gap_and_last" (save the last iteration as 'params.pkl' and save every snapshot_gap iteration separately)
        - "gap_overwrite" (same as gap but overwrites the last saved snapshot)
        - "none" (do not save snapshots).
      snapshot_gap (int): Gap between snapshot iterations.
    """
    self.snapshot_dir = snapshot_dir
    self.snapshot_mode = snapshot_mode
    self.snapshot_gap = snapshot_gap
    pass

    if snapshot_mode == 'gap_overwrite' and snapshot_gap <= 1:
      raise ValueError('Snapshot_gap must be > 1 when using snapshot_mode="gap_overwrite". Use  snapshot_mode="last" '
                       'to snapshot after every iteration.')

    if snapshot_mode == 'last' and snapshot_gap != 1:
      raise ValueError('snapshot_gap should be set to 1 if using snapshot_mode="last". Did you mean to use '
                       'snapshot_mode="gap"?')

    pathlib.Path(snapshot_dir).mkdir(parents = True, exist_ok = True)
    pass

  def save_snapshot(self, itr, params):
    """
    Save the parameters if at the right iteration.

    Args:
      itr (int): Number of iterations.
      params (obj): Content of snapshot to be saved.

    Raises:
      ValueError: If snapshot_mode is not one of "all", "last", "gap", "gap_overwrite", "gap_and_last", or "none".
    """
    file_name = None
    if self.snapshot_mode == 'all':
      file_name = os.path.join(self.snapshot_dir, 'itr_%d.pkl' % itr)

    elif self.snapshot_mode == 'gap_overwrite':
      if itr % self.snapshot_gap == 0:
        file_name = os.path.join(self.snapshot_dir, 'params.pkl')

    elif self.snapshot_mode == 'last':
      file_name = os.path.join(self.snapshot_dir, 'params.pkl')

    elif self.snapshot_mode == 'gap':
      if itr % self.snapshot_gap == 0:
        file_name = os.path.join(self.snapshot_dir, 'itr_%d.pkl' % itr)

    elif self.snapshot_mode == 'gap_and_last':
      if itr % self.snapshot_gap == 0:
        file_name = os.path.join(self.snapshot_dir, 'itr_%d.pkl' % itr)

      file_name_last = os.path.join(self.snapshot_dir, 'params.pkl')
      with open(file_name_last, 'wb') as file:
        cloudpickle.dump(params, file)

    elif self.snapshot_mode == 'none':
      pass

    else:
      raise ValueError('Invalid snapshot mode {}'.format(self.snapshot_mode))

    if file_name:
      with open(file_name, 'wb') as file:
        cloudpickle.dump(params, file)

  def load(self, load_dir, itr = 'last'):
    """
    Load one snapshot of parameters from disk.

    Args:
      load_dir (str): Directory of the cloudpickle file to resume experiment from.
      itr (int or string): Iteration to load. Can be an integer, 'last' or 'first'.

    Returns:
      dict: Loaded snapshot.

    Raises:
      ValueError: If itr is neither an integer nor one of ("last", "first").
      FileNotFoundError: If the snapshot file is not found in load_dir.
      NotAFileError: If the snapshot exists but is not a file.
    """
    if isinstance(itr, int) or itr.isdigit():
      load_from_file = os.path.join(load_dir, 'itr_{}.pkl'.format(itr))
    else:
      if itr not in ('last', 'first'):
        raise ValueError("itr should be an integer or 'last' or 'first'")

      load_from_file = os.path.join(load_dir, 'params.pkl')
      if not os.path.isfile(load_from_file):
        files = [f for f in os.listdir(load_dir) if f.endswith('.pkl')]
        if not files:
          raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), '*.pkl file in', load_dir)

        files.sort(key = _extract_snapshot_itr)
        load_from_file = files[0] if itr == 'first' else files[-1]
        load_from_file = os.path.join(load_dir, load_from_file)

    if not os.path.isfile(load_from_file):
      raise NotAFileError('File not existing: ', load_from_file)

    with open(load_from_file, 'rb') as file:
      return cloudpickle.load(file)


def _extract_snapshot_itr(filename: str) -> int:
  """
  Extracts the integer itr from a filename.

  Args:
    filename(str): The snapshot filename.

  Returns:
    int: The snapshot as an integer.
  """
  base = os.path.splitext(filename)[0]
  digits = base.split('itr_')[1]

  return int(digits)


class NotAFileError(Exception):
  """
  Raise when the snapshot is not a file.
  """
