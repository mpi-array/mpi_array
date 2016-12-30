"""
=================================
The :mod:`mpi_array.array` Module
=================================

.. currentmodule:: mpi_array.array

Defines multi-dimensional distributed array class.

Classes and Functions
=====================

.. autosummary::
   :toctree: generated/

   ndarray - multi-dimensional distributed array.

"""
from __future__ import absolute_import
from .license import license as _license, copyright as _copyright

import mpi_array as _mpi_array
import mpi_array.logging as _logging  # noqa: E402,F401
import numpy as _np  # noqa: E402,F401
import mpi4py.MPI as _mpi  # noqa: E402,F401

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _mpi_array.__version__


class ndarray(_np.ndarray):
    """
    Multi-dimensional distributed array.
    """
    pass


def zeros(*args, **kwargs):
    pass