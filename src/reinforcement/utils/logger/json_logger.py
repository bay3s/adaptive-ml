import json

from .key_value_logger_abc import KeyValueLoggerABC


class JSONLogger(KeyValueLoggerABC):

  def __init__(self, filename):
    self.file = open(filename, 'at')

  def write_key_values(self, kvs):
    for k, v in sorted(kvs.items()):
      if hasattr(v, 'dtype'):
        v = v.tolist()
        kvs[k] = float(v)
    self.file.write(json.dumps(kvs) + '\n')
    self.file.flush()

  def close(self):
    self.file.close()
