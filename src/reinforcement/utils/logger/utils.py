import os
import sys

from .json_logger import JSONLogger
from .tensorboard_logger import TensorBoardLogger
from .csv_logger import CSVLogger
from .readable_logger import ReadableLogger


def make_output_format(format, ev_dir, log_suffix=''):
    os.makedirs(ev_dir, exist_ok=True)
    if format == 'stdout':
        return ReadableLogger(sys.stdout)
    elif format == 'log':
        return ReadableLogger(osp.join(ev_dir, 'log%s.txt' % log_suffix))
    elif format == 'json':
        return JSONLogger(osp.join(ev_dir, 'progress%s.json' % log_suffix))
    elif format == 'csv':
        return CSVLogger(osp.join(ev_dir, 'progress%s.csv' % log_suffix))
    elif format == 'tensorboard':
        return TensorBoardLogger(osp.join(ev_dir, 'tb%s' % log_suffix))
    else:
        raise ValueError('Unknown format specified: %s' % (format,))
