from .key_value_logger_abc import KeyValueLoggerABC
from .sequence_logger_abc import SequenceLoggerABC


class ReadableLogger(KeyValueLoggerABC, SequenceLoggerABC):

  def __init__(self, file_path):
    self.file = open(file_path, 'at')
    self.own_file = True

  def write_key_values(self, key_values: dict) -> None:
    key2str = {}
    for (key, val) in sorted(key_values.items()):
      if isinstance(val, float):
        valstr = '%-8.3g' % (val,)
      else:
        valstr = str(val)
      key2str[self._truncate(key)] = self._truncate(valstr)

    # Find max widths
    if len(key2str) == 0:
      print('WARNING: tried to write empty key-value dict')
      return
    else:
      keywidth = max(map(len, key2str.keys()))
      valwidth = max(map(len, key2str.values()))

    # Write out the data
    dashes = '-' * (keywidth + valwidth + 7)
    lines = [dashes]
    for (key, val) in sorted(key2str.items()):
      lines.append('| %s%s | %s%s |' % (
        key,
        ' ' * (keywidth - len(key)),
        val,
        ' ' * (valwidth - len(val)),
      ))
    lines.append(dashes)
    self.file.write('\n'.join(lines) + '\n')

    # Flush the output to the file
    self.file.flush()

  def write_sequence(self, seq):
    for arg in seq:
      self.file.write(arg)
    self.file.write('\n')
    self.file.flush()

  def _truncate(self, s):
    return s[:20] + '...' if len(s) > 23 else s

  def close(self):
    if self.own_file:
      self.file.close()
