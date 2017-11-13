"""
The :mod:`mpi_array.benchmarks.__main__` Module
===============================================

Command line running of benchmarking.
Execute::

   python -m mpi_array.benchmarks --help

for command line options.

"""
from __future__ import absolute_import
from ..license import license as _license, copyright as _copyright, version as _version

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()

if __name__ == '__main__':
    import sys as _sys
    import logging as _builtin_logging
    from . import benchmark as _benchmark
    from .. import logging as _logging

    _sys.path.pop(0)
    _logging.initialise_loggers(["mpi_array.benchmarks.benchmark"], log_level=_builtin_logging.INFO)
    _benchmark.run_main(_sys.argv[1:])
