"""
========================================
The :mod:`mpi_array.array_test` Module
========================================

.. currentmodule:: mpi_array.array_test

Module defining :mod:`mpi_array.ndarray` unit-tests.
Execute as::

   python -m mpi_array.array_test


Classes
=======

.. autosummary::
   :toctree: generated/

   NdarrayTest - :obj:`unittest.TestCase` for :mod:`mpi_array.ndarray` functions.


"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright
import mpi_array.unittest as _unittest
import mpi_array.logging as _logging  # noqa: E402,F401
import mpi_array as _mpi_array

import numpy as _np  # noqa: E402,F401


__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _mpi_array.__version__


class NdarrayTest(_unittest.TestCase):

    """
    :obj:`unittest.TestCase` for :mod:`mpi_array.ndarray` functions.
    """


_unittest.main(__name__)


__all__ = [s for s in dir() if not s.startswith('_')]
