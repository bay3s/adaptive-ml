from .key_value_logger_abc import KeyValueLoggerABC


class CSVLogger(KeyValueLoggerABC):
  def __init__(self, filename):
    self.file = open(filename, 'a+t')
    self.keys = []
    self.sep = ','

  def write_key_values(self, kvs):
    # Add our current row to the history
    extra_keys = kvs.keys() - self.keys
    if extra_keys:
      self.keys.extend(extra_keys)
      self.file.seek(0)
      lines = self.file.readlines()
      self.file.seek(0)
      for (i, k) in enumerate(self.keys):
        if i > 0:
          self.file.write(',')
        self.file.write(k)
      self.file.write('\n')
      for line in lines[1:]:
        self.file.write(line[:-1])
        self.file.write(self.sep * len(extra_keys))
        self.file.write('\n')
    for (i, k) in enumerate(self.keys):
      if i > 0:
        self.file.write(',')
      v = kvs.get(k)
      if v is not None:
        self.file.write(str(v))
    self.file.write('\n')
    self.file.flush()

  def close(self):
    self.file.close()
