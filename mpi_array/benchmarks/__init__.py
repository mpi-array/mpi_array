"""
=======================================
The :mod:`mpi_array.benchmarks` Package
=======================================

Runtime benchmarking.

Modules
=======

.. autosummary::
   :toctree: generated/

   benchmark - Functions and classes for finding/recording benchmarks.
   bench_creation - Benchmark array creation.
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
