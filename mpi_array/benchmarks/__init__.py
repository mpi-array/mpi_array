"""
=======================================
The :mod:`mpi_array.benchmarks` Package
=======================================

Runtime benchmarking, in the style of `airspeed velocity (asv) <https://asv.readthedocs.io/>`_.
Run the benchmarks (here those of :mod:`mpi_array.benchmarks.bench_update`, on 4 processes)
with::

   mpirun -n 4 python -m mpi4py -m mpi_array.benchmarks -t Wtime \\
       -f 'benchmarks.bench_update' -o results.json -b benchmarks.json

The :samp:`-q` (quick) option runs each benchmark once, :samp:`--help` lists the options.

Modules
=======

.. autosummary::
   :toctree: generated/

   benchmark - Functions and classes for finding/recording benchmarks.
   bench_creation - Benchmark array creation.
   bench_ufunc - Benchmarks for ufuncs.
   bench_update - Benchmarks for copyto and halo updates.
   core - Core utilities for benchmark implementations.
   utils - Miscellaneous helper utilities.
"""
from __future__ import absolute_import
from ..license import license as _license, copyright as _copyright, version as _version

from . import benchmark  # noqa: E402,F401
from . import bench_creation  # noqa: E402,F401

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()

__all__ = [s for s in dir() if not s.startswith('_')]
