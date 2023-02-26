import sys
import os.path as osp
import joblib
from collections import defaultdict

from .key_value_logger_abc import KeyValueLoggerABC
from .sequence_logger_abc import SequenceLoggerABC
from .readable_logger import ReadableLogger

from .constants import (
  DEBUG,
  INFO,
  WARN,
  ERROR,
  DISABLED
)


def logkv(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.
    """
    Logger.CURRENT.log_key_value(key, val)


def logkv_mean(key, val):
    """
    The same as logkv(), but if called many times, values averaged.
    """
    Logger.CURRENT.log_key_value_mean(key, val)


def logkvs(d):
    """
    Log a dictionary of key-value pairs
    """
    for (k, v) in d.items():
        logkv(k, v)


def dumpkvs():
    """
    Write diagnostics from the current iteration.

    level: int. (see logger.py docs) If the global logger level is higher than
                the level argument here, don't print to stdout.
    """
    Logger.CURRENT.dump_key_values()


def getkvs():
    return Logger.CURRENT.name2val


def log(*args, level=INFO):
    """
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
    Logger.CURRENT.log(*args, level=level)


def debug(*args):
    log(*args, level=DEBUG)


def info(*args):
    log(*args, level=INFO)


def warn(*args):
    log(*args, level=WARN)


def error(*args):
    log(*args, level=ERROR)


def set_level(level):
    """
    Set logging threshold on current logger.
    """
    Logger.CURRENT.set_level(level)


def get_dir():
    """
    Get directory that log files are being written to.
    will be None if there is no output directory (i.e., if you didn't call start)
    """
    return Logger.CURRENT.get_dir()


def save_itr_params(*args):
    return Logger.CURRENT.save_itr_params(*args)


record_tabular = logkv
dump_tabular = dumpkvs


class Logger:

    DEFAULT = None
    CURRENT = None

    def __init__(self, dir, output_formats, snapshot_mode='last', snapshot_gap=1):
        self.name2val = defaultdict(float)  # values this iteration
        self.name2cnt = defaultdict(int)
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats
        self.snapshot_mode = snapshot_mode
        self.snapshot_gap = snapshot_gap

    def log_key_value(self, key, val):
        self.name2val[key] = val

    def log_key_value_mean(self, key, val):
        if val is None:
            self.name2val[key] = None
            return
        oldval, cnt = self.name2val[key], self.name2cnt[key]
        self.name2val[key] = oldval*cnt/(cnt+1) + val/(cnt+1)
        self.name2cnt[key] = cnt + 1

    def dump_key_values(self):
        if self.level == DISABLED:
            return

        for fmt in self.output_formats:
            if isinstance(fmt, KeyValueLoggerABC):
                fmt.write_key_values(self.name2val)

        self.name2val.clear()
        self.name2cnt.clear()

    def log(self, *args, level=INFO):
        if self.level <= level:
            self._do_log(args)

    # Configuration
    # ----------------------------------------
    def set_level(self, level):
        self.level = level

    def get_dir(self):
        return self.dir

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        for fmt in self.output_formats:
            if isinstance(fmt, SequenceLoggerABC):
                fmt.write_sequence(map(str, args))

    def save_itr_params(self, itr, params):
        if self.dir:
            if self.snapshot_mode == 'all':
                file_name = osp.join(self.dir, 'itr_%d.pkl' % itr)
                joblib.dump(params, file_name, compress=3)
            elif self.snapshot_mode == 'last':
                # override previous params
                file_name = osp.join(self.dir, 'params.pkl')
                joblib.dump(params, file_name, compress=3)
            elif self.snapshot_mode == "gap":
                if itr % self.snapshot_gap == 0:
                    itr_file_name = osp.join(self.dir, 'itr.txt')
                    with open(itr_file_name, 'w') as itr_file:
                        itr_file.write(str(itr))
                    file_name = osp.join(self.dir, 'itr_%d.pkl' % itr)
                    joblib.dump(params, file_name, compress=3)
            elif self.snapshot_mode == 'last_gap':
                if itr % self.snapshot_gap == 0:
                    file_name = osp.join(self.dir, 'params.pkl')
                    joblib.dump(params, file_name, compress=3)
            elif self.snapshot_mode == 'none':
                pass
            else:
                raise NotImplementedError


Logger.DEFAULT = Logger.CURRENT = Logger(dir=None, output_formats=[ReadableLogger(sys.stdout)])
