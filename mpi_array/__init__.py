"""
==============================
The :mod:`mpi_array` Package
==============================

Python package for multi-dimensional distributed arrays like
:obj:`numpy.ndarray`.


Classes and Functions
=====================

.. autosummary::
   :toctree: generated/

Attributes
==========


"""
from __future__ import absolute_import  # noqa: E402,F401
from . import rtd as _rtd  # noqa: E402,F401
from .license import license as _license, copyright as _copyright
import pkg_resources as _pkg_resources

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _pkg_resources.resource_string("mpi_array", "version.txt").decode()

from . import array  # noqa: E402,F401
from .array import ndarray, zeros  # noqa: E402,F401

__all__ = [s for s in dir() if not s.startswith('_')]
