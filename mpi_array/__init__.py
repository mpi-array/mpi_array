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

import sys as _sys
from . import rtd as _rtd  # noqa: E402,F401
from .license import license as _license, copyright as _copyright, version as _version

__author__ = "Shane J. Latham"
__license__ = _license()
__copyright__ = _copyright()
__version__ = _version()

_thismodule = _sys.modules[__name__]

from . import globale as _globale  # noqa: E402,F401
gndarray = _globale.gndarray

from . import globale_creation as _creation  # noqa: E402,F401
for s in _creation.__all__:
    setattr(_thismodule, s, getattr(_creation, s))

__all__ = [s for s in dir() if not s.startswith('_')]
