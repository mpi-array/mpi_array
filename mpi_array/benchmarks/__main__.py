"""
The :mod:`mpi_array.benchmarks.__main__` Module
===============================================

Command line running of benchmarking.
Execute::

   python -m mpi_array.benchmarks --help

for command line options.

"""
from __future__ import absolute_import
import sys as _sys
from ..license import license as _license, copyright as _copyright, version as _version
from . import benchmark as _benchmark

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()

if __name__ == '__main__':
    _sys.path.pop(0)
    _benchmark.run_main(_sys.argv)
