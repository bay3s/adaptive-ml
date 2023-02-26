import os
import os.path as osp
import time
import datetime
import tempfile

from .constants import (
  LOG_OUTPUT_FORMATS,
  LOG_OUTPUT_FORMATS_MPI,
  DEBUG,
  INFO,
  WARN,
  ERROR
)

from .utils import make_output_format
from .logger import Logger


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


class profiler:
    """
    Usage:
    with logger.profiler("interesting_scope"):
        code
    """
    def __init__(self, n):
        self.n = "wait_" + n

    def __enter__(self):
        self.t1 = time.time()

    def __exit__(self, type, value, traceback):
        Logger.CURRENT.name2val[self.n] += time.time() - self.t1


def profile(n):
    """
    Usage:
    @profile("my_func")
    def my_func(): code
    """
    def decorator_with_name(func):
        def func_wrapper(*args, **kwargs):
            with profiler(n):
                return func(*args, **kwargs)

        return func_wrapper

    return decorator_with_name


def configure(directory=None, format_strs=None, snapshot_mode= 'last', snapshot_gap=1):
    if directory is None:
        directory = os.getenv('ADAPTIVE_ML_LOGDIR')

    if directory is None:
        data_time_formatted = datetime.datetime.now().strftime("adaptive-ml-%Y-%m-%d-%H-%M-%S-%f")
        directory = osp.join(tempfile.gettempdir(), data_time_formatted)

    assert isinstance(directory, str)
    os.makedirs(directory, exist_ok=True)

    log_suffix = ''
    from mpi4py import MPI
    rank = MPI.COMM_WORLD.Get_rank()
    if rank > 0:
        log_suffix = "-rank%03i" % rank

    if format_strs is None:
        strs, strs_mpi = os.getenv('ADAPTIVE_ML_LOG_FORMAT'), os.getenv('ADAPTIVE_ML_LOG_FORMAT_MPI')
        format_strs = strs_mpi if rank > 0 else strs
        if format_strs is not None:
            format_strs = format_strs.split(',')
        else:
            format_strs = LOG_OUTPUT_FORMATS_MPI if rank > 0 else LOG_OUTPUT_FORMATS

    output_formats = [make_output_format(f, directory, log_suffix) for f in format_strs]

    Logger.CURRENT = Logger(dir=directory, output_formats=output_formats, snapshot_mode=snapshot_mode, snapshot_gap=snapshot_gap)
    log('Logging to %s' % directory)


def reset():
    if Logger.CURRENT is not Logger.DEFAULT:
        Logger.CURRENT.close()
        Logger.CURRENT = Logger.DEFAULT
        log('Reset logger')

