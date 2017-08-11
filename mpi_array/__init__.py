"""
============================
The :mod:`mpi_array` Package
============================

Python package for multi-dimensional distributed arrays
(
`Partitioned Global Address Space <https://en.wikipedia.org/wiki/Partitioned_global_address_space>`_
)
with :obj:`numpy.ndarray`-like API.


Classes and Functions
=====================

.. autosummary::
   :toctree: generated/

Attributes
==========


"""
from __future__ import absolute_import  # noqa: E402,F401
from . import rtd as _rtd  # noqa: E402,F401
from .license import license as _license, copyright as _copyright, version as _version

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()

from .globale import gndarray  # noqa: E402,F401
from .globale_creation import copy, empty, empty_like  # noqa: E402,F401
from .globale_creation import ones, ones_like, zeros, zeros_like  # noqa: E402,F401

__all__ = [s for s in dir() if not s.startswith('_')]
